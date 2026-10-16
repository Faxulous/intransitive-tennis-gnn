from dataclasses import dataclass
from math import comb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
from utils.helpers import load_player_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── KM iid point model (Klaassen & Magnus 2001) ──
# Reference C = median(pA + pB) for ATP; WTA values differ — always compute from data
_KM_C_REF = {'Hard': 1.279, 'Clay': 1.240, 'Grass': 1.328}
_KM_CACHE = {}  # (C_rounded, best_of) -> interp1d

def _km_game(p):  # P(server wins game) given serve point prob p
    q = 1 - p
    return p**4 * (1 + 4*q + 10*q**2) + 20*p**3*q**3 * p**2 / (p**2 + q**2)

def _km_set(pA, pB):  # P(A wins set) via DP over game scores
    gA, gB = _km_game(pA), _km_game(pB)
    memo = {}
    def dp(a, b, srv):
        if (a, b, srv) in memo: return memo[(a, b, srv)]
        if a == 6 and b == 6:  # Tiebreak
            p = (pA + 1 - pB) / 2; q = 1 - p
            r = sum(comb(6+k, k) * p**7 * q**k for k in range(6)) + comb(12,6) * p**6 * q**6 * p**2 / (p**2 + q**2)
        elif a >= 6 and a - b >= 2: r = 1.0
        elif b >= 6 and b - a >= 2: r = 0.0
        else:
            pw = gA if srv else (1 - gB)
            r = pw * dp(a+1, b, not srv) + (1-pw) * dp(a, b+1, not srv)
        memo[(a, b, srv)] = r
        return r
    return dp(0, 0, True)

def _km_match(pA, pB, bo=3):  # P(A wins match)
    s = _km_set(pA, pB)
    return 3*s**2 - 2*s**3 if bo == 3 else 10*s**3 - 15*s**4 + 6*s**5

def _km_inversion_table(C, bo, n=2000):
    """Build/cache interpolation: match_prob -> pA_implied serve prob."""
    key = (round(C, 3), bo)
    if key not in _KM_CACHE:
        lo, hi = max(0.35, C - 0.85), min(0.85, C - 0.35)
        grid = np.linspace(lo, hi, n)
        mp = np.array([_km_match(pa, C - pa, bo) for pa in grid])
        _KM_CACHE[key] = interp1d(mp, grid, bounds_error=False, fill_value=(lo, hi))
    return _KM_CACHE[key]


@dataclass
class WindowStack:
    timestamp: int
    matches: torch.Tensor  # (n, 2) - query_edges
    results: torch.Tensor  # (n,)   - y_true (binary)
    windows: torch.Tensor  # (n,)   - window_idx (int)
    tourney_level: str
    best_of: torch.Tensor
    set_results: torch.Tensor # (n_sets,) - set outcomes
    match_ids: torch.Tensor
    game_pct: torch.Tensor = None  # Game-based dominance
    set_pct: torch.Tensor = None   # Set-based dominance
    point_pct: torch.Tensor = None # Point-based dominance (total pts won / total pts)
    serve_pct: torch.Tensor = None # Serve-based dominance (relative serve win %)
    serve_surprise: torch.Tensor = None  # Actual serve % minus KM-implied (winner perspective)
    point_surprise: torch.Tensor = None  # Actual point% minus Pin-implied point% (winner perspective)
    break_pct: torch.Tensor = None       # Break point conversion dominance (relative)
    break_surprise: torch.Tensor = None  # Actual bp conversion minus KM-implied (directional)
    game_error: torch.Tensor = None     # A_game_pct residual from linear fit on PSA_ip
    surface: str = ""  # New: surface identifier (e.g. "Hard", "Clay", "Grass")
    rounds: list = None  # per-match Round strings (e.g. '1st Round','The Final') for (level,round) tq lookup
    market_probs: torch.Tensor = None  # Market probabilities for decorrelation (player A win probability)
    # Per-player raw stats (A/B perspective, NOT winner's) for H2H edge features
    a_serve_pct: torch.Tensor = None   # A's serve pts won / total serve pts
    b_serve_pct: torch.Tensor = None
    a_return_pct: torch.Tensor = None  # A's return pts won / total return pts
    b_return_pct: torch.Tensor = None
    a_bp_pct: torch.Tensor = None      # A's break point conversion
    b_bp_pct: torch.Tensor = None
    # Serve decomposition (per-player, for dynamic node features)
    a_1st_in_pct: torch.Tensor = None   # A's 1st serve in %
    b_1st_in_pct: torch.Tensor = None
    a_1st_win_pct: torch.Tensor = None  # A's 1st serve win %
    b_1st_win_pct: torch.Tensor = None
    a_2nd_win_pct: torch.Tensor = None  # A's 2nd serve win %
    b_2nd_win_pct: torch.Tensor = None
    a_ace_rate: torch.Tensor = None     # A's aces / total serve pts
    b_ace_rate: torch.Tensor = None
    a_df_rate: torch.Tensor = None      # A's DFs / total serve pts
    b_df_rate: torch.Tensor = None

    def __post_init__(self):
        # Update validation to include set_pct and market_probs
        n = self.matches.size(0)
        if not all(tensor.size(0) == n for tensor in 
                  [self.results, self.windows, self.best_of, self.set_results, self.match_ids] 
                  if tensor is not None):
            raise ValueError("All tensors must have same first dimension")
        if self.game_pct is not None and self.game_pct.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.set_pct is not None and self.set_pct.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.point_pct is not None and self.point_pct.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.serve_pct is not None and self.serve_pct.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.serve_surprise is not None and self.serve_surprise.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.point_surprise is not None and self.point_surprise.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.break_pct is not None and self.break_pct.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.break_surprise is not None and self.break_surprise.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.game_error is not None and self.game_error.size(0) != n:
            raise ValueError("All tensors must have same first dimension")
        if self.market_probs is not None and self.market_probs.size(0) != n:
            raise ValueError("All tensors must have same first dimension")

    @property
    def max_window(self) -> int:
        return int(self.windows.max())

    @property
    def min_window(self) -> int:
        if len(self.windows) == 0:
            return -1  # or raise an exception
        return int(self.windows.min())



def merge_window_stacks(window_stacks_list):
    """Merge window stacks, ignoring tournament type differences"""
    if not window_stacks_list:
        return None
        
    # Use latest timestamp and first window's tourney_level
    merged = WindowStack(
        timestamp=window_stacks_list[-1].timestamp,  # Use latest timestamp
        matches=torch.cat([ws.matches for ws in window_stacks_list]),
        results=torch.cat([ws.results for ws in window_stacks_list]),
        windows=torch.cat([ws.windows for ws in window_stacks_list]),
        tourney_level=window_stacks_list[0].tourney_level,  # Use first window's type
        best_of=torch.cat([ws.best_of for ws in window_stacks_list]),
        set_results=torch.cat([ws.set_results for ws in window_stacks_list]),
        match_ids=torch.cat([ws.match_ids for ws in window_stacks_list]),
        game_pct=torch.cat([ws.game_pct for ws in window_stacks_list if ws.game_pct is not None]) if window_stacks_list[0].game_pct is not None else None,
        set_pct=torch.cat([ws.set_pct for ws in window_stacks_list if ws.set_pct is not None]) if window_stacks_list[0].set_pct is not None else None,
        point_pct=torch.cat([ws.point_pct for ws in window_stacks_list if ws.point_pct is not None]) if window_stacks_list[0].point_pct is not None else None,
        serve_pct=torch.cat([ws.serve_pct for ws in window_stacks_list if ws.serve_pct is not None]) if window_stacks_list[0].serve_pct is not None else None,
        serve_surprise=torch.cat([ws.serve_surprise for ws in window_stacks_list if ws.serve_surprise is not None]) if window_stacks_list[0].serve_surprise is not None else None,
        point_surprise=torch.cat([ws.point_surprise for ws in window_stacks_list if ws.point_surprise is not None]) if window_stacks_list[0].point_surprise is not None else None,
        break_pct=torch.cat([ws.break_pct for ws in window_stacks_list if ws.break_pct is not None]) if window_stacks_list[0].break_pct is not None else None,
        break_surprise=torch.cat([ws.break_surprise for ws in window_stacks_list if ws.break_surprise is not None]) if window_stacks_list[0].break_surprise is not None else None,
        game_error=torch.cat([ws.game_error for ws in window_stacks_list if ws.game_error is not None]) if window_stacks_list[0].game_error is not None else None,
        surface=window_stacks_list[0].surface,
        market_probs=torch.cat([ws.market_probs for ws in window_stacks_list if ws.market_probs is not None]) if window_stacks_list[0].market_probs is not None else None,
        a_serve_pct=torch.cat([ws.a_serve_pct for ws in window_stacks_list if ws.a_serve_pct is not None]) if window_stacks_list[0].a_serve_pct is not None else None,
        b_serve_pct=torch.cat([ws.b_serve_pct for ws in window_stacks_list if ws.b_serve_pct is not None]) if window_stacks_list[0].b_serve_pct is not None else None,
        a_return_pct=torch.cat([ws.a_return_pct for ws in window_stacks_list if ws.a_return_pct is not None]) if window_stacks_list[0].a_return_pct is not None else None,
        b_return_pct=torch.cat([ws.b_return_pct for ws in window_stacks_list if ws.b_return_pct is not None]) if window_stacks_list[0].b_return_pct is not None else None,
        a_bp_pct=torch.cat([ws.a_bp_pct for ws in window_stacks_list if ws.a_bp_pct is not None]) if window_stacks_list[0].a_bp_pct is not None else None,
        b_bp_pct=torch.cat([ws.b_bp_pct for ws in window_stacks_list if ws.b_bp_pct is not None]) if window_stacks_list[0].b_bp_pct is not None else None,
        a_1st_in_pct=torch.cat([ws.a_1st_in_pct for ws in window_stacks_list if ws.a_1st_in_pct is not None]) if window_stacks_list[0].a_1st_in_pct is not None else None,
        b_1st_in_pct=torch.cat([ws.b_1st_in_pct for ws in window_stacks_list if ws.b_1st_in_pct is not None]) if window_stacks_list[0].b_1st_in_pct is not None else None,
        a_1st_win_pct=torch.cat([ws.a_1st_win_pct for ws in window_stacks_list if ws.a_1st_win_pct is not None]) if window_stacks_list[0].a_1st_win_pct is not None else None,
        b_1st_win_pct=torch.cat([ws.b_1st_win_pct for ws in window_stacks_list if ws.b_1st_win_pct is not None]) if window_stacks_list[0].b_1st_win_pct is not None else None,
        a_2nd_win_pct=torch.cat([ws.a_2nd_win_pct for ws in window_stacks_list if ws.a_2nd_win_pct is not None]) if window_stacks_list[0].a_2nd_win_pct is not None else None,
        b_2nd_win_pct=torch.cat([ws.b_2nd_win_pct for ws in window_stacks_list if ws.b_2nd_win_pct is not None]) if window_stacks_list[0].b_2nd_win_pct is not None else None,
        a_ace_rate=torch.cat([ws.a_ace_rate for ws in window_stacks_list if ws.a_ace_rate is not None]) if window_stacks_list[0].a_ace_rate is not None else None,
        b_ace_rate=torch.cat([ws.b_ace_rate for ws in window_stacks_list if ws.b_ace_rate is not None]) if window_stacks_list[0].b_ace_rate is not None else None,
        a_df_rate=torch.cat([ws.a_df_rate for ws in window_stacks_list if ws.a_df_rate is not None]) if window_stacks_list[0].a_df_rate is not None else None,
        b_df_rate=torch.cat([ws.b_df_rate for ws in window_stacks_list if ws.b_df_rate is not None]) if window_stacks_list[0].b_df_rate is not None else None,
        rounds=sum([ws.rounds for ws in window_stacks_list if ws.rounds is not None], []) if window_stacks_list[0].rounds is not None else None,
    )
    return merged


@dataclass
class SurfaceMetaStack:
    expired: list[WindowStack]
    graph: list[WindowStack]
    train: list[WindowStack]
    val: list[WindowStack]
    test: list[WindowStack]

    def min_window(self) -> int:
        """Retrieve the current minimum window from the test stack."""
        first_window_in_test = self.test[0].windows
        win_idxs = first_window_in_test.unique()
        if win_idxs.numel() != 1:
            raise ValueError("More than one unique window idx in test[0].windows.")
        return win_idxs.item()

    def cascade_left(self):
        """Cascade 0th WindowStack from test → val → train → graph (append to the end)."""
        if self.test:
            self.val.append(self.test.pop(0))
        if self.val:
            self.train.append(self.val.pop(0))
        if self.train:
            self.graph.append(self.train.pop(0))
        if self.expired: # @todo: more sophisticated length before expiring windows
            self.expired.append(self.graph.pop(0))


def create_meta_stacks(split_surface_data, gender):
    """
    Converts nested dict values into SurfaceMetaStack objects by splitting DataFrames by window_idx.

    Args:
        split_surface_data (dict): The input nested dictionary with structure {key1: {key2: DataFrame}}.
        gender: The gender parameter for data conversion.

    Returns:
        dict: A dictionary of {key1: SurfaceMetaStack}.
    """
    # Load player data once and reuse it
    player_data = load_player_data(gender)
    
    # Create a fast lookup mapping for player names to indices
    player_name_to_idx = {name: data['index'] for name, data in player_data.items()}
    
    meta_stacks = {}

    for key1, inner_dict in split_surface_data.items():
        # Initialize a defaultdict to hold WindowStacks by key2
        window_stacks = defaultdict(list)

        for key2, df in inner_dict.items():
            # Group DataFrame by window_idx and convert each group to a WindowStack
            grouped = df.groupby("window_idx")
            for window_idx, window_df in grouped:
                window_stack = convert_df_to_tensor(window_df, player_name_to_idx)
                window_stacks[key2].append(window_stack)

        # Create SurfaceMetaStack using the grouped WindowStacks
        meta_stacks[key1] = SurfaceMetaStack(
            [None], # empty list for expired stack
            window_stacks['graph'],
            window_stacks['train'],
            window_stacks['val'],
            window_stacks['test']
        )

    return meta_stacks


def split_df_by_surface(df, graph_pct, train_pct, val_pct):
    """
    Split data chronologically first (global), then organize by surface.
    This ensures temporal integrity across all surfaces.
    """
    # Global chronological split
    df_sorted = df.sort_values(['window_idx', 'match_id'])
    total_rows = len(df_sorted)
    
    graph_end = int(total_rows * graph_pct)
    train_end = graph_end + int(total_rows * train_pct) 
    val_end = train_end + int(total_rows * val_pct)
    
    # Split chronologically first
    splits = {
        'graph': df_sorted.iloc[:graph_end],
        'train': df_sorted.iloc[graph_end:train_end], 
        'val': df_sorted.iloc[train_end:val_end],
        'test': df_sorted.iloc[val_end:]
    }
    
    # Then organize by surface
    surface_splits = {}
    for surface in df['Surface'].unique():
        surface_splits[surface] = {
            split_name: split_df[split_df['Surface'] == surface] 
            for split_name, split_df in splits.items()
        }
    
    return surface_splits


def _name_candidates(full):  # Generate abbreviated name variants, most specific first
    parts = full.split()
    if len(parts) < 2:
        return [full]
    surname, first_parts = parts[-1], parts[:-1]
    init = first_parts[0][0] + '.'
    # Two-letter disambiguation: "Andrey" -> "An.", "Zhizhen" -> "Zh."
    init2 = first_parts[0][:2] + '.' if len(first_parts[0]) >= 2 else init
    candidates = []
    # Most specific first — disambiguated two-letter initial
    if init2 != init:
        candidates.append(surname + ' ' + init2)  # Kuznetsov An., Zhang Zh.
    candidates.append(surname + ' ' + init)  # Nadal R.
    # Multi-word surname: "Van De Zandschulp B."
    if len(parts) > 2:
        candidates.append(' '.join(parts[1:]) + ' ' + init)
        candidates.append(parts[-2] + '-' + parts[-1] + ' ' + init)
    # Compound first name initials: "Jo-Wilfried" -> J.W., J-W.
    if len(first_parts) >= 2 or '-' in first_parts[0]:
        sub = first_parts[0].replace('-', ' ').split() if '-' in first_parts[0] else first_parts
        compound = '.'.join(s[0] for s in sub) + '.'
        candidates.append(surname + ' ' + compound)
        candidates.append(surname + ' ' + '-'.join(s[0] for s in sub) + '.')
    # 3+ word surnames: "Juan Martin Del Potro" -> "Del Potro J.M."
    if len(parts) > 3:
        sub = parts[:-2]
        compound = '.'.join(s[0] for s in sub) + '.'
        candidates.append(' '.join(parts[-2:]) + ' ' + compound)
        candidates.append(' '.join(parts[-2:]) + ' ' + init)
        candidates.append(' '.join(parts[-3:]) + ' ' + init)
    # Intermediate word as surname: "Jaume Antoni Munar Clar" -> "Munar J."
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1):
            candidates.append(parts[i] + ' ' + init)
            if init2 != init:
                candidates.append(parts[i] + ' ' + init2)
    # Title-case variants for particles (de -> De)
    for c in list(candidates):
        candidates.append(c.title())
    return candidates

_NAME_CACHE = {}   # full_name -> idx, built per player_name_to_idx id
_PREFIX_MAP = {}   # "Surname F" -> player_name (unique) or None (ambiguous)

def _build_name_cache(player_name_to_idx):
    """Cache is tied to a specific player_name_to_idx dict."""
    cache_key = id(player_name_to_idx)
    if cache_key not in _NAME_CACHE:
        _NAME_CACHE[cache_key] = {}
    return _NAME_CACHE[cache_key]

def _build_prefix_map(player_name_to_idx):
    """Build 'Surname F' -> full_name map for fuzzy matching multi-initial entries."""
    cache_key = id(player_name_to_idx)
    if cache_key not in _PREFIX_MAP:
        pm = {}
        for pname in player_name_to_idx:
            p = pname.split()
            if len(p) >= 2:
                prefix = ' '.join(p[:-1]) + ' ' + p[-1][0]  # "Wolf J.J." -> "Wolf J"
                if prefix in pm:
                    pm[prefix] = None  # Ambiguous
                else:
                    pm[prefix] = pname
        _PREFIX_MAP[cache_key] = pm
    return _PREFIX_MAP[cache_key]

# Manual aliases for players whose player_data entry can't be reached by _name_candidates.
# Reasons: typo in source CSV, stage-name vs full-name mismatch, compound first names.
_NAME_ALIASES = {
    'Michelle Larcher De Brito': 'Larcherde Brito M.',  # typo: missing space in player_data
    'Tomas Barrios Vera':        'Barrios Vera M.T.',   # stage name (full: Marcelo Tomas)
    'Yifan Xu':                  'Xu Y.F.',             # compound first name Yi-Fan
}


def convert_df_to_tensor(df, player_name_to_idx):

    Aname_col, Bname_col = 'Aname', 'Bname'
    cache = _build_name_cache(player_name_to_idx)
    pm = _build_prefix_map(player_name_to_idx)

    def lookup(name):
        if name in cache:
            return cache[name]
        if name in _NAME_ALIASES:  # manual override for edge cases
            idx = player_name_to_idx.get(_NAME_ALIASES[name], -1)
            cache[name] = idx
            return idx
        idx = player_name_to_idx.get(name, -1)
        if idx == -1:
            for c in _name_candidates(name):
                idx = player_name_to_idx.get(c, -1)
                if idx != -1:
                    break
        if idx == -1:  # Prefix match: "Wolf J." -> "Wolf J.J."
            for c in _name_candidates(name):
                p = c.split()
                if len(p) >= 2:
                    prefix = ' '.join(p[:-1]) + ' ' + p[-1][0]
                    match = pm.get(prefix)
                    if match:
                        idx = player_name_to_idx[match]
                        break
        cache[name] = idx
        return idx

    A_idx = df[Aname_col].map(lookup).to_numpy()
    B_idx = df[Bname_col].map(lookup).to_numpy()

    # Filter out matches with unmapped players
    valid_matches_mask = (A_idx != -1) & (B_idx != -1)
    if not all(valid_matches_mask):
        import warnings
        n_skip = int(sum(~valid_matches_mask))
        unmapped = sorted(set(df.loc[A_idx == -1, Aname_col].unique()) | set(df.loc[B_idx == -1, Bname_col].unique()))
        warnings.warn(f"{n_skip} matches skipped — unmapped: {unmapped}", stacklevel=2)
        
        # Apply the mask to df and other arrays
        df = df[valid_matches_mask].reset_index(drop=True)
        A_idx = A_idx[valid_matches_mask]
        B_idx = B_idx[valid_matches_mask]

    # Optimized DataFrame expansion using vectorized operations
    _e = lambda: torch.empty(0, dtype=torch.float, device=device)
    _el = lambda: torch.empty(0, dtype=torch.long, device=device)
    def _empty_stack():
        return WindowStack(0, torch.empty((0, 2), dtype=torch.long), _el(), _el(), "",
            _el(), _el(), _el(), game_pct=_e(), set_pct=_e(), point_pct=_e(),
            serve_pct=_e(), serve_surprise=_e(), point_surprise=_e(),
            break_pct=_e(), break_surprise=_e(), game_error=_e(), market_probs=_e(),
            a_serve_pct=_e(), b_serve_pct=_e(), a_return_pct=_e(), b_return_pct=_e(),
            a_bp_pct=_e(), b_bp_pct=_e(),
            a_1st_in_pct=_e(), b_1st_in_pct=_e(), a_1st_win_pct=_e(), b_1st_win_pct=_e(),
            a_2nd_win_pct=_e(), b_2nd_win_pct=_e(), a_ace_rate=_e(), b_ace_rate=_e(),
            a_df_rate=_e(), b_df_rate=_e())

    if df.empty:
        return _empty_stack()

    # Calculate total sets for each match and ensure no NaN/negative values
    total_sets = (df['Asets'] + df['Bsets']).fillna(0).astype(int)
    total_sets = np.maximum(total_sets, 0)  # Ensure non-negative

    # Filter out matches with zero sets
    valid_sets_mask = total_sets > 0
    if not any(valid_sets_mask):
        return _empty_stack()
    
    # Apply the mask to both dataframe and total_sets
    df_valid = df[valid_sets_mask].reset_index(drop=True)
    total_sets_valid = total_sets[valid_sets_mask]
    
    # Create expanded arrays using numpy repeat  
    expanded_indices = np.repeat(range(len(df_valid)), total_sets_valid)
    set_numbers = np.concatenate([np.arange(1, sets + 1) for sets in total_sets_valid])
    
    # Create expanded dataframe more efficiently
    expanded_df = df_valid.iloc[expanded_indices].copy().reset_index(drop=True)
    expanded_df['set_num'] = set_numbers
    
    # Truly vectorized set winner calculation
    set_winners = []
    for match_idx, num_sets in enumerate(total_sets_valid):
        match_row = df_valid.iloc[match_idx]
        for set_num in range(1, num_sets + 1):
            set_winners.append(1 if match_row[f'A{set_num}'] > match_row[f'B{set_num}'] else 0)
    
    expanded_df['set_winner'] = set_winners

    A_idx_expanded = expanded_df[Aname_col].map(lookup).to_numpy()
    B_idx_expanded = expanded_df[Bname_col].map(lookup).to_numpy()

    query_edges = torch.from_numpy(np.vstack([A_idx_expanded, B_idx_expanded])).long().to(device).T
    y_true = torch.from_numpy(expanded_df['Awin'].to_numpy()).long().to(device)
    set_results = torch.from_numpy(expanded_df['set_winner'].to_numpy()).long().to(device)
    window_idx = torch.from_numpy(expanded_df['window_idx'].to_numpy()).long().to(device)
    match_ids = torch.from_numpy(expanded_df['match_id'].to_numpy()).long().to(device)
    
    game_pct = torch.from_numpy(
        np.where(expanded_df['Awin'], expanded_df['A_game_pct'], 1 - expanded_df['A_game_pct'])
    ).float().to(device)
    set_pct = torch.from_numpy(
        np.where(expanded_df['Awin'], expanded_df['A_set_pct'], 1 - expanded_df['A_set_pct'])
    ).float().to(device)
    # ── Shared retirement mask: winner didn't reach required sets ──
    aw = expanded_df['Awin'].to_numpy(dtype=bool)
    bo_arr = expanded_df['Best of'].to_numpy(dtype=int)
    asets = expanded_df['Asets'].to_numpy(dtype=float)
    bsets = expanded_df['Bsets'].to_numpy(dtype=float)
    retired = np.maximum(asets, bsets) < (bo_arr // 2 + 1)
    extreme_pin = (expanded_df['PSA_ip'] < 0.05) | (expanded_df['PSA_ip'] > 0.95)

    # ── Point pct: winner's share of total points ──
    a_pts = expanded_df['A_total_pts'].to_numpy(dtype=float)
    b_pts = expanded_df['B_total_pts'].to_numpy(dtype=float)
    total = a_pts + b_pts
    with np.errstate(invalid='ignore', divide='ignore'):
        a_point_pct = np.where(total > 0, a_pts / total, np.nan)
    winner_point_pct = np.where(aw, a_point_pct, 1 - a_point_pct)
    winner_point_pct[retired] = np.nan
    point_pct = torch.from_numpy(winner_point_pct).float().to(device)

    # ── Point surprise: actual point% - Pin-implied point% + 0.5 (directional) ──
    pt_surp_arr = np.full(len(expanded_df), np.nan)
    if 'pA_implied' in expanded_df.columns:
        pA_imp_pt = expanded_df['pA_implied'].to_numpy(dtype=float)
        pB_imp_pt = expanded_df['pB_implied'].to_numpy(dtype=float)
        a_imp_pt = (pA_imp_pt + (1 - pB_imp_pt)) / 2  # A's expected point fraction
        w_imp_pt = np.where(aw, a_imp_pt, 1 - a_imp_pt)
        pt_surp_arr = winner_point_pct - w_imp_pt + 0.5
        pt_surp_arr[retired | extreme_pin] = np.nan
    point_surprise = torch.from_numpy(pt_surp_arr).float().to(device)

    # ── Serve pct: relative serve dominance ──
    a_srv_won = expanded_df['A_1st_won'].to_numpy(dtype=float) + expanded_df['A_2nd_won'].to_numpy(dtype=float)
    a_srv_tot = expanded_df['A_1st_won_total'].to_numpy(dtype=float) + expanded_df['A_2nd_won_total'].to_numpy(dtype=float)
    b_srv_won = expanded_df['B_1st_won'].to_numpy(dtype=float) + expanded_df['B_2nd_won'].to_numpy(dtype=float)
    b_srv_tot = expanded_df['B_1st_won_total'].to_numpy(dtype=float) + expanded_df['B_2nd_won_total'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        a_srv_pct = np.where(a_srv_tot > 0, a_srv_won / a_srv_tot, np.nan)
        b_srv_pct = np.where(b_srv_tot > 0, b_srv_won / b_srv_tot, np.nan)
        srv_sum = a_srv_pct + b_srv_pct
        a_serve_dom = np.where(srv_sum > 0, a_srv_pct / srv_sum, np.nan)
    winner_serve = np.where(aw, a_serve_dom, 1 - a_serve_dom)
    winner_serve[retired] = np.nan
    serve_pct = torch.from_numpy(winner_serve).float().to(device)

    # ── Serve surprise: actual serve % minus KM-implied (floored at 0.5) ──
    surp_arr = np.full(len(expanded_df), np.nan)
    if 'pA_implied' in expanded_df.columns:
        pA_imp = expanded_df['pA_implied'].to_numpy(dtype=float)
        pB_imp = expanded_df['pB_implied'].to_numpy(dtype=float)
        surp_arr = 0.5 + np.where(aw, a_srv_pct - pA_imp - (b_srv_pct - pB_imp),
                                       b_srv_pct - pB_imp - (a_srv_pct - pA_imp))
        surp_arr[retired | extreme_pin] = np.nan
    serve_surprise = torch.from_numpy(np.clip(surp_arr, 0.5, 1.0)).float().to(device)

    # ── Break point pct: winner's bp conversion / (winner's + loser's) ──
    a_bp_won = expanded_df['A_bp_won'].to_numpy(dtype=float)
    a_bp_tot = expanded_df['A_bp_total'].to_numpy(dtype=float)
    b_bp_won = expanded_df['B_bp_won'].to_numpy(dtype=float)
    b_bp_tot = expanded_df['B_bp_total'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        a_bp_conv = np.where(a_bp_tot > 0, a_bp_won / a_bp_tot, np.nan)
        b_bp_conv = np.where(b_bp_tot > 0, b_bp_won / b_bp_tot, np.nan)
        bp_sum = a_bp_conv + b_bp_conv
        a_bp_dom = np.where(bp_sum > 0, a_bp_conv / bp_sum, np.nan)
    winner_bp = np.where(aw, a_bp_dom, 1 - a_bp_dom)
    winner_bp[retired] = np.nan
    break_pct = torch.from_numpy(winner_bp).float().to(device)

    # ── Break surprise: actual bp conversion minus KM-implied + 0.5 (directional) ──
    bp_surp_arr = np.full(len(expanded_df), np.nan)
    if 'pA_implied' in expanded_df.columns:
        # KM-implied break conversion: prob of breaking = 1 - opp_serve_prob
        a_bp_exp = 1 - pB_imp  # A's expected break rate against B
        b_bp_exp = 1 - pA_imp  # B's expected break rate against A
        w_bp_surp = np.where(aw, a_bp_conv - a_bp_exp, b_bp_conv - b_bp_exp)
        l_bp_surp = np.where(aw, b_bp_conv - b_bp_exp, a_bp_conv - a_bp_exp)
        bp_surp_arr = np.clip(0.5 + (w_bp_surp - l_bp_surp), 0.0, 1.0)  # directional, clipped
        bp_surp_arr[retired | extreme_pin] = np.nan
    break_surprise = torch.from_numpy(bp_surp_arr).float().to(device)

    # ── Game error: precomputed A_game_pct residual from linear fit on PSA_ip ──
    if 'game_error' in expanded_df.columns:
        game_error = torch.from_numpy(expanded_df['game_error'].to_numpy(dtype=float)).float().to(device)
    else:
        game_error = None

    best_of = torch.from_numpy(expanded_df['Best of'].to_numpy()).long().to(device)
    
    # ── Raw per-player stats (A/B perspective) for H2H edge features ──
    a_ret = expanded_df['A_return_won'].to_numpy(dtype=float)
    a_ret_tot = expanded_df['A_return_total'].to_numpy(dtype=float)
    b_ret = expanded_df['B_return_won'].to_numpy(dtype=float)
    b_ret_tot = expanded_df['B_return_total'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        a_ret_pct = np.where(a_ret_tot > 0, a_ret / a_ret_tot, np.nan)
        b_ret_pct = np.where(b_ret_tot > 0, b_ret / b_ret_tot, np.nan)
    # Mask retirements for all raw stats
    for arr in [a_srv_pct, b_srv_pct, a_ret_pct, b_ret_pct, a_bp_conv, b_bp_conv]:
        arr[retired] = np.nan

    a_serve_pct_t = torch.from_numpy(a_srv_pct).float().to(device)
    b_serve_pct_t = torch.from_numpy(b_srv_pct).float().to(device)
    a_return_pct_t = torch.from_numpy(a_ret_pct).float().to(device)
    b_return_pct_t = torch.from_numpy(b_ret_pct).float().to(device)
    a_bp_pct_t = torch.from_numpy(a_bp_conv).float().to(device)
    b_bp_pct_t = torch.from_numpy(b_bp_conv).float().to(device)

    # ── Serve decomposition: 1st_in%, 1st_win%, 2nd_win%, ace_rate, df_rate ──
    a_1st_in = expanded_df['A_1st_in'].to_numpy(dtype=float)
    a_1st_tot = expanded_df['A_1st_total'].to_numpy(dtype=float)
    b_1st_in = expanded_df['B_1st_in'].to_numpy(dtype=float)
    b_1st_tot = expanded_df['B_1st_total'].to_numpy(dtype=float)
    a_1st_won = expanded_df['A_1st_won'].to_numpy(dtype=float)
    b_1st_won = expanded_df['B_1st_won'].to_numpy(dtype=float)
    a_2nd_won = expanded_df['A_2nd_won'].to_numpy(dtype=float)
    a_2nd_tot = expanded_df['A_2nd_won_total'].to_numpy(dtype=float)  # Total 2nd serve points
    b_2nd_won = expanded_df['B_2nd_won'].to_numpy(dtype=float)
    b_2nd_tot = expanded_df['B_2nd_won_total'].to_numpy(dtype=float)
    a_aces = expanded_df['A_aces'].to_numpy(dtype=float)
    b_aces = expanded_df['B_aces'].to_numpy(dtype=float)
    a_dfs = expanded_df['A_df'].to_numpy(dtype=float)
    b_dfs = expanded_df['B_df'].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        _a1in = np.where(a_1st_tot > 0, a_1st_in / a_1st_tot, np.nan)
        _b1in = np.where(b_1st_tot > 0, b_1st_in / b_1st_tot, np.nan)
        _a1w  = np.where(a_1st_in > 0, a_1st_won / a_1st_in, np.nan)
        _b1w  = np.where(b_1st_in > 0, b_1st_won / b_1st_in, np.nan)
        _a2w  = np.where(a_2nd_tot > 0, a_2nd_won / a_2nd_tot, np.nan)
        _b2w  = np.where(b_2nd_tot > 0, b_2nd_won / b_2nd_tot, np.nan)
        _aace = np.where(a_srv_tot > 0, a_aces / a_srv_tot, np.nan)
        _bace = np.where(b_srv_tot > 0, b_aces / b_srv_tot, np.nan)
        _adf  = np.where(a_srv_tot > 0, a_dfs / a_srv_tot, np.nan)
        _bdf  = np.where(b_srv_tot > 0, b_dfs / b_srv_tot, np.nan)
    for arr in [_a1in, _b1in, _a1w, _b1w, _a2w, _b2w, _aace, _bace, _adf, _bdf]:
        arr[retired] = np.nan
    _to = lambda x: torch.from_numpy(x).float().to(device)

    # Extract market odds and convert to probabilities for decorrelation training
    market_probs = torch.from_numpy(expanded_df['PSA_ip'].to_numpy()).float().to(device)
    # Use original df for metadata to preserve original logic
    timestamp = int(pd.to_datetime(df['Date'].iloc[-1]).timestamp()) if not df.empty else 0

    # Extract tourney_level from DataFrame - use expanded_df since it has the actual data we're using
    tourney_level = expanded_df['tourney_level'].iloc[0] if not expanded_df.empty else ""
    rounds = expanded_df['Round'].tolist() if not expanded_df.empty else []  # per-match round strings
    
    # Use original df for surface to maintain consistency with original logic
    surface = df['Surface'].iloc[0] if 'Surface' in df.columns and not df.empty else ""

    return WindowStack(
        timestamp,
        query_edges,
        y_true,
        window_idx,
        tourney_level,
        best_of,
        set_results,
        match_ids,
        game_pct=game_pct,
        set_pct=set_pct,
        point_pct=point_pct,
        serve_pct=serve_pct,
        serve_surprise=serve_surprise,
        point_surprise=point_surprise,
        break_pct=break_pct,
        break_surprise=break_surprise,
        game_error=game_error,
        surface=surface,
        market_probs=market_probs,
        a_serve_pct=a_serve_pct_t, b_serve_pct=b_serve_pct_t,
        a_return_pct=a_return_pct_t, b_return_pct=b_return_pct_t,
        a_bp_pct=a_bp_pct_t, b_bp_pct=b_bp_pct_t,
        a_1st_in_pct=_to(_a1in), b_1st_in_pct=_to(_b1in),
        a_1st_win_pct=_to(_a1w), b_1st_win_pct=_to(_b1w),
        a_2nd_win_pct=_to(_a2w), b_2nd_win_pct=_to(_b2w),
        a_ace_rate=_to(_aace), b_ace_rate=_to(_bace),
        a_df_rate=_to(_adf), b_df_rate=_to(_bdf),
        rounds=rounds,
    )
