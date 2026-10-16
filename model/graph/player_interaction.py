from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix, dok_matrix
from torch_geometric_signed_directed.utils import in_out_degree
from utils.helpers import load_player_data
from model.graph.ia_compute import calculate_surface_intransitivity, calculate_match_local_intransitivity_vectorized, calculate_match_local_intransitivity_vectorized_extended
from model.graph.tq_buckets import TIER_MAP, FALLBACK_TIER, TIERS, get_tier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PlayerInteractionMatrix:
    """
    Player Interaction Matrix (PIM) class for storing and updating interaction matrices for each surface.
    """
    gender: str # Defines which gender to use (men, women, both)
    granularity: str # Defines which granularity to use (match, set, game)
    surfaces: Dict[str, Dict[str, float]] = field(default_factory=dict)  # No default
    lambda_param: float = 0.001
    tournament_quality: Dict[str, float] = field(default_factory=dict)  # No default
    kappa: float = 0.0  # Prior strength for Beta(kappa/2, kappa/2) shrinkage (Efron & Morris 1975)
    max_edge_age: float = 0.0  # Max edge age in years; 0 = no expiry
    min_dom_pctile: float = 0.0  # Percentile [0,1] of |dominance-0.5| below which matches are skipped; 0 = all
    min_edge_prob: float = 0.0  # Min |posterior| to keep edge in graph; 0 = no filter
    node_feature_matrix: Optional[torch.Tensor] = None
    # Don't touch these
    alpha: float = field(init=False)
    n_nodes: int = field(init=False)
    interaction_matrices: Dict[str, dok_matrix] = field(init=False)
    surface_graphs: Dict[str, Dict[str, Optional[torch.Tensor]]] = field(init=False)
    cumulative_weight: Dict[str, dok_matrix] = field(init=False)
    weighted_sum: Dict[str, dok_matrix] = field(init=False)


    def __post_init__(self):
        # Validate tournament_quality keys: must be the 4 tier labels (GS, Finals, 1000, 500)
        if set(self.tournament_quality.keys()) != set(TIERS):
            raise ValueError(f"tournament_quality keys must be exactly {TIERS}, got {list(self.tournament_quality.keys())}")

        # Create node feature matrix that includes players filtered by gender
        if self.node_feature_matrix is None:
            self.node_feature_matrix = PlayerInteractionMatrix._create_node_feature_matrix(gender=self.gender)
        self.n_nodes = self.node_feature_matrix.shape[0]
        
        # Get surfaces
        filtered_surfaces = self.surfaces.copy()
        
        # Initialize empty interaction matrices
        self.interaction_matrices = {}
        self.last_update_matrices = {}
        self.cumulative_weight = {}
        self.weighted_sum = {}
        
        # Use lil_matrix for faster initialization
        for surface in filtered_surfaces:
            # These can start completely empty
            self.interaction_matrices[surface] = lil_matrix((self.n_nodes, self.n_nodes), dtype=np.float32).tocsr().todok()
            self.last_update_matrices[surface] = lil_matrix((self.n_nodes, self.n_nodes), dtype=np.float32).tocsr().todok()
            
            # For these, we need the small constants - use efficient broadcasting
            cw = lil_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
            ws = lil_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
            
            # Only set diagonal elements to small constants (since we only need non-zero values for actual interactions)
            cw.setdiag(np.full(self.n_nodes, 1e-8))
            ws.setdiag(np.full(self.n_nodes, 0.0))  # 0-centered: neutral prior = 0 evidence
            
            self.cumulative_weight[surface] = cw.tocsr().todok()
            self.weighted_sum[surface] = ws.tocsr().todok()
        
        # H2H per-player stat tracker: cross-surface running averages {(a,b): {'a_serve': [sum,count], ...}}
        self.h2h_stats = {}
        # Per-player running stats: 8 metrics × [sum, count] = 16 cols
        # [srv, ret, bp, 1st_in, 1st_win, 2nd_win, ace, df] × [sum, n]
        self.player_stats = np.zeros((self.n_nodes, 16))

        # Initialize empty surface graphs
        self.surface_graphs = {
            surface: {"X_real": None, "X_img": None, "edge_index": None, "edge_weight": None}
            for surface in filtered_surfaces
        }
        

    def _update_pair(self, surface: str, i: int, j: int, current_time: float,
                    match_dominance: float, correlation: float = 1.0):
        """
        0-centered Bayesian update for players i and j on the given surface.

        match_dominance is in [-0.5, 0.5]: positive = i dominated, 0 = neutral.
        Stored edge weight = |posterior|; direction by which key (i,j) vs (j,i) is nonzero.
        Decay pulls posterior toward 0 (no edge) rather than 0.5 (ambiguous).
        """
        # Ensure ordered pair; negate dominance if swapped
        if i > j:
            i, j = j, i
            match_dominance = -match_dominance

        last_time = self.last_update_matrices[surface].get((i, j), current_time)
        delta_years = max(0, current_time - last_time) / (365 * 24 * 60 * 60)
        decay = np.exp(np.clip(-self.lambda_param * delta_years, -700, 0))

        # Prior centered at 0; kappa pseudo-observations with 0 evidence
        old_weight = self.cumulative_weight[surface].get((i, j), max(self.kappa, 1e-8))
        old_evidence = self.weighted_sum[surface].get((i, j), 0.0)

        decayed_weight = decay * old_weight
        decayed_evidence = decay * old_evidence

        updated_weight = decayed_weight + correlation
        updated_evidence = decayed_evidence + correlation * match_dominance

        # Posterior in [-0.5, 0.5]
        posterior = updated_evidence / updated_weight if updated_weight > 1e-10 else 0.0
        posterior = max(-0.5, min(0.5, posterior))

        # Clear both directions, store dominant direction with magnitude
        self.interaction_matrices[surface][(i, j)] = 0.0
        self.interaction_matrices[surface][(j, i)] = 0.0
        if posterior > 0:
            self.interaction_matrices[surface][(i, j)] = posterior  # i dominates j
        elif posterior < 0:
            self.interaction_matrices[surface][(j, i)] = -posterior  # j dominates i

        self.cumulative_weight[surface][(i, j)] = updated_weight
        self.cumulative_weight[surface][(j, i)] = updated_weight
        self.weighted_sum[surface][(i, j)] = updated_evidence
        self.weighted_sum[surface][(j, i)] = -updated_evidence

        self.last_update_matrices[surface][(i, j)] = current_time
        self.last_update_matrices[surface][(j, i)] = current_time

    def update_interaction_matrix(self, surface: str, window):
        # Windows carry one row per set, so a match updates the pair once per set played
        results = window.results.cpu().numpy()
        matches = window.matches.cpu().numpy()
        timestamp = window.timestamp
        
        # Choose dominance signal: games, sets, or binary win/loss
        if self.granularity == "games":
            dominance_pcts = window.game_pct.cpu().numpy()
        elif self.granularity == "sets":
            dominance_pcts = window.set_pct.cpu().numpy()
        elif self.granularity == "points":
            dominance_pcts = window.point_pct.cpu().numpy()
        elif self.granularity == "serve":
            dominance_pcts = window.serve_pct.cpu().numpy()
        elif self.granularity == "serve_surprise":
            if window.serve_surprise is None:
                raise ValueError("serve_surprise requires pA_implied/pB_implied columns, which this package does not build")
            dominance_pcts = window.serve_surprise.cpu().numpy()
        elif self.granularity == "point_surprise":
            if window.point_surprise is None:
                raise ValueError("point_surprise requires pA_implied/pB_implied columns, which this package does not build")
            dominance_pcts = window.point_surprise.cpu().numpy()
        elif self.granularity == "break":
            dominance_pcts = window.break_pct.cpu().numpy()
        elif self.granularity == "break_surprise":
            if window.break_surprise is None:
                raise ValueError("break_surprise requires pA_implied/pB_implied columns, which this package does not build")
            dominance_pcts = window.break_surprise.cpu().numpy()
        elif self.granularity == "binary":
            dominance_pcts = np.ones(len(results))  # 1.0 for winner always
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")
            
        tourney_level = window.tourney_level
        # Per-match tq weight: tourney_level -> tier (GS/Finals/1000/500) -> weight. Fallback to '500' for unseen levels.
        tq_weight = self.tournament_quality[get_tier(tourney_level)]
        tq_weights = np.full(len(results), tq_weight)
        # For binary, filter on game_pct (binary has no dominance variation)
        filter_pcts = window.game_pct.cpu().numpy() if self.granularity == "binary" else dominance_pcts

        # Compute percentile-based dominance threshold (granularity-agnostic)
        dom_threshold = 0.0
        if self.min_dom_pctile > 0:
            deviations = np.abs(filter_pcts - 0.5)
            valid = deviations[~np.isnan(deviations)]
            if len(valid) > 0:
                dom_threshold = np.percentile(valid, self.min_dom_pctile * 100)

        winner_indices = np.where(results == 1, matches[:, 0], matches[:, 1])
        loser_indices = np.where(results == 1, matches[:, 1], matches[:, 0])

        for idx, (i, j) in enumerate(zip(winner_indices.astype(int), loser_indices.astype(int))):
            current_game_pct = dominance_pcts[idx]

            # Skip matches with missing dominance data
            if np.isnan(current_game_pct):
                continue

            # Skip matches below dominance percentile threshold
            if dom_threshold > 0 and abs(filter_pcts[idx] - 0.5) < dom_threshold:
                continue

            # 0-center: shift from [0,1] to [-0.5, 0.5]
            centered_dominance = current_game_pct - 0.5

            self._update_pair(
                surface, i, j, timestamp,
                match_dominance=centered_dominance,
                correlation=1.0
            )

            # Update correlated surfaces' matrices
            for surf_prime in self.surfaces:
                if surf_prime == surface:
                    continue
                correlation = self.surfaces[surface].get(surf_prime, 0.0)
                if correlation > 0:
                    # Scale cross-surface influence by per-match tournament-tier weight
                    effective_correlation = correlation * tq_weights[idx]

                    self._update_pair(
                        surf_prime, i, j, timestamp,
                        match_dominance=centered_dominance,
                        correlation=effective_correlation
                    )

        # Update H2H + per-player stats (A/B perspective, not winner/loser)
        has_stats = (window.a_serve_pct is not None and window.a_return_pct is not None
                     and window.a_bp_pct is not None)
        if has_stats:
            a_srv = window.a_serve_pct.cpu().numpy()
            b_srv = window.b_serve_pct.cpu().numpy()
            a_ret = window.a_return_pct.cpu().numpy()
            b_ret = window.b_return_pct.cpu().numpy()
            a_bp = window.a_bp_pct.cpu().numpy()
            b_bp = window.b_bp_pct.cpu().numpy()
            # Serve decomposition (may be None if data unavailable)
            _cpu = lambda t: t.cpu().numpy() if t is not None else None
            a_1in, b_1in = _cpu(window.a_1st_in_pct), _cpu(window.b_1st_in_pct)
            a_1w, b_1w = _cpu(window.a_1st_win_pct), _cpu(window.b_1st_win_pct)
            a_2w, b_2w = _cpu(window.a_2nd_win_pct), _cpu(window.b_2nd_win_pct)
            a_ace, b_ace = _cpu(window.a_ace_rate), _cpu(window.b_ace_rate)
            a_df, b_df = _cpu(window.a_df_rate), _cpu(window.b_df_rate)
            has_decomp = a_1in is not None

            for idx in range(len(matches)):
                a, b = int(matches[idx, 0]), int(matches[idx, 1])
                stats = [a_srv[idx], b_srv[idx], a_ret[idx], b_ret[idx], a_bp[idx], b_bp[idx]]
                if any(np.isnan(s) for s in stats):
                    continue
                self._update_h2h_stats(a, b, stats)
                # Per-player running averages: 8 metrics × [sum, count] = 16 cols
                row_a = [a_srv[idx], 1, a_ret[idx], 1, a_bp[idx], 1]
                row_b = [b_srv[idx], 1, b_ret[idx], 1, b_bp[idx], 1]
                if has_decomp:
                    # Per-metric NaN handling: add [val, 1] if valid, [0, 0] if NaN
                    for va, vb in [(a_1in,b_1in),(a_1w,b_1w),(a_2w,b_2w),(a_ace,b_ace),(a_df,b_df)]:
                        va_ok = not np.isnan(va[idx])
                        vb_ok = not np.isnan(vb[idx])
                        row_a += [va[idx], 1] if va_ok else [0, 0]
                        row_b += [vb[idx], 1] if vb_ok else [0, 0]
                else:
                    row_a += [0]*10
                    row_b += [0]*10
                self.player_stats[a] += row_a
                self.player_stats[b] += row_b

    def _update_h2h_stats(self, a, b, stats):
        """Update cross-surface running-average H2H stats for pair (a, b)."""
        keys = ['a_serve', 'b_serve', 'a_return', 'b_return', 'a_bp', 'b_bp']
        pair = (a, b)  # Ordered by original A/B, not sorted
        if pair not in self.h2h_stats:
            self.h2h_stats[pair] = {k: [0.0, 0.0] for k in keys}
        entry = self.h2h_stats[pair]
        for k, val in zip(keys, stats):
            entry[k][0] += val   # Running sum
            entry[k][1] += 1.0   # Count

    def create_surface_graph(self, surface: str, current_time: float = None):
        """Convert to CSR before creating graph, pruning stale edges."""
        # Prune edges older than max_edge_age before graph creation
        if current_time is not None and self.max_edge_age > 0:
            cutoff = current_time - self.max_edge_age * 365 * 24 * 60 * 60
            im = self.interaction_matrices[surface]
            lu = self.last_update_matrices[surface]
            stale = [(i, j) for (i, j), w in im.items() if w != 0 and lu.get((i, j), 0) < cutoff]
            for i, j in stale:
                im[(i, j)] = 0

        # Prune edges with weak dominance (stored weights are |posterior|, close to 0 = inconclusive)
        if self.min_edge_prob > 0:
            im = self.interaction_matrices[surface]
            weak = [(i, j) for (i, j), w in im.items() if w != 0 and w < self.min_edge_prob]
            for i, j in weak:
                im[(i, j)] = 0

        # Convert to CSR format
        interaction_matrix = self.interaction_matrices[surface].tocsr()
        
        # Convert sparse matrix to PyTorch sparse tensor
        interaction_matrix_coo = interaction_matrix.tocoo()
        
        # Create mask for non-zero weights
        mask = interaction_matrix_coo.data != 0
        filtered_rows = interaction_matrix_coo.row[mask]
        filtered_cols = interaction_matrix_coo.col[mask]
        filtered_data = interaction_matrix_coo.data[mask]
        
        edge_index = torch.tensor(
            np.vstack((filtered_rows, filtered_cols)),  # Use filtered rows/cols
            dtype=torch.long,
            device=device
        )
        edge_weight = torch.tensor(filtered_data, dtype=torch.float32, device=device)  # Use filtered data
        # Calculate in/out degrees and add them to node features (adds 2 features)
        X_degrees = in_out_degree(edge_index, size=len(self.node_feature_matrix)).to(device)
        # Normalize static + degree features together (original behaviour)
        X_static = F.normalize(
            torch.cat([self.node_feature_matrix.to(device), X_degrees.to(device)], dim=1),
            p=2, dim=0, eps=1e-12
        )
        # Dynamic per-player stats: 8 metrics (0-centered, no data = 0)
        # [srv, ret, bp, 1st_in, 1st_win, 2nd_win, ace, df]
        counts = self.player_stats[:, 1::2]  # 8 count columns
        sums = self.player_stats[:, 0::2]    # 8 sum columns
        with np.errstate(invalid='ignore', divide='ignore'):
            avgs = np.where(counts > 0, sums / counts, 0.5)
        avgs -= 0.5  # 0-center: no data = 0, above avg = positive
        X_dynamic = torch.from_numpy(avgs).float().to(device)
        # Concatenate: 6 normalized static + 8 dynamic = 14
        X_real = torch.cat([X_static, X_dynamic], dim=1)
        X_img = torch.zeros_like(X_real).to(device)
        
        self.surface_graphs[surface] = {
            "X_real": X_real,
            "X_img": X_img,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }

    def remove_from_interaction_matrix(self, surface, window):
        # Convert to numpy arrays once, outside the loop
        results = window.results.cpu().numpy()
        matches = window.matches.cpu().numpy()
        winner_indices = np.where(
            results == 1,
            matches[:, 0],
            matches[:, 1]
        )
        loser_indices = np.where(
            results == 1,
            matches[:, 1],
            matches[:, 0]
        )
        window_time = window.timestamp
        
        # Get all last update times at once
        pairs = list(zip(winner_indices.astype(int), loser_indices.astype(int)))
        last_updates = np.array([self.last_update_matrices[surface].get((i,j), 0) for i,j in pairs])
        
        # Find which pairs need zeroing
        to_zero = last_updates <= window_time
        
        # Zero out all relevant pairs at once
        for i, j in np.array(pairs)[to_zero]:
            self.interaction_matrices[surface][(i, j)] = 0
            self.interaction_matrices[surface][(j, i)] = 0
            
            # By deleting the keys, we reset them to the default prior state for the next update
            for mat in [self.cumulative_weight[surface], self.weighted_sum[surface], self.last_update_matrices[surface]]:
                for pair in [(i, j), (j, i)]:
                    try:
                        del mat[pair]
                    except KeyError:
                        pass


    def init_surface_graphs(self, current_time=None):
        for surface in self.surfaces:
            self.create_surface_graph(surface, current_time)


    def create_matrix_copy(self):
        """Create a deep copy of the interaction matrices and associated data structures."""
        copy_pim = PlayerInteractionMatrix(
            gender=self.gender,
            granularity=self.granularity,
            surfaces=self.surfaces.copy(),
            lambda_param=self.lambda_param,
            tournament_quality=self.tournament_quality.copy(),
            kappa=self.kappa,
            max_edge_age=self.max_edge_age,
            min_dom_pctile=self.min_dom_pctile,
            min_edge_prob=self.min_edge_prob,
            node_feature_matrix=self.node_feature_matrix.clone()
        )
        
        copy_pim.n_nodes = self.n_nodes
        
        # Deep copy the sparse matrices
        for surface in self.surfaces:
            copy_pim.interaction_matrices[surface] = self.interaction_matrices[surface].copy()
            copy_pim.last_update_matrices[surface] = self.last_update_matrices[surface].copy()
            copy_pim.cumulative_weight[surface] = self.cumulative_weight[surface].copy()
            copy_pim.weighted_sum[surface] = self.weighted_sum[surface].copy()
        # Deep copy H2H stats (cross-surface, not per-surface)
        copy_pim.h2h_stats = {
            k: {sk: list(sv) for sk, sv in v.items()}
            for k, v in self.h2h_stats.items()
        }
        copy_pim.player_stats = self.player_stats.copy()

        return copy_pim

    # Helper function to update PIM with data from specified stages
    def update_pim_from_stacks(self, meta_stacks, stages_to_include):
        """Updates a PIM instance with data from specified stages."""
        latest_time = None
        for surface, stack in meta_stacks.items():
            for stage in stages_to_include:
                if hasattr(stack, stage):
                    stage_windows = getattr(stack, stage)
                    if stage_windows:
                        for window in stage_windows:
                            self.update_interaction_matrix(surface, window)
                            if latest_time is None or window.timestamp > latest_time:
                                latest_time = window.timestamp
        self.init_surface_graphs(latest_time)

    

    def detect_cycle_closing_matches(self, surface: str, query_edges: torch.Tensor, 
                                   min_prob_threshold: float = 0.6, current_time: float = None):
        """
        Efficiently detect which query matches would close 3-cycles using vectorized operations.
        
        Args:
            surface (str): The surface to check
            query_edges (torch.Tensor): Shape (n_matches, 2), each row is [player_i, player_j]
            min_prob_threshold (float): Minimum edge probability to consider strong
            current_time (float): Current timestamp for recency calculations
            
        Returns:
            List[Dict]: Cycle information for each cycle-closing match
        """
        if surface not in self.interaction_matrices:
            return []
        
        # Convert to dense numpy for efficient operations
        matrix = self.interaction_matrices[surface].toarray()
        query_edges_np = query_edges.cpu().numpy().astype(int)
        
        # Create strong edge mask
        strong_mask = matrix >= min_prob_threshold
        
        cycle_results = []
        
        for match_idx, (a, b) in enumerate(query_edges_np):
            cycle_info = None
            
            # Check A->X->B cycles (A beats someone who beats B)
            a_dominates = strong_mask[a, :]  # Who does A beat?
            beats_b = strong_mask[:, b]      # Who beats B?
            intermediates_axb = np.where(a_dominates & beats_b)[0]
            
            if len(intermediates_axb) > 0:
                # Found A->X->B cycle, betting on A closes it
                x = intermediates_axb[0]  # Take first intermediate
                cycle_probs = [matrix[a, x], matrix[x, b]]
                cycle_info = {
                    'bet_on_player_idx': a,
                    'cycle_path': [a, x, b],
                    'cycle_probs': cycle_probs,
                }
            else:
                # Check B->X->A cycles (B beats someone who beats A)  
                b_dominates = strong_mask[b, :]
                beats_a = strong_mask[:, a]
                intermediates_bxa = np.where(b_dominates & beats_a)[0]
                
                if len(intermediates_bxa) > 0:
                    # Found B->X->A cycle, betting on B closes it
                    x = intermediates_bxa[0]
                    cycle_probs = [matrix[b, x], matrix[x, a]]
                    cycle_info = {
                        'bet_on_player_idx': b,
                        'cycle_path': [b, x, a],
                        'cycle_probs': cycle_probs,
                    }
            
            if cycle_info:
                # Calculate cycle statistics and recency
                min_prob = min(cycle_info['cycle_probs'])
                avg_prob = np.mean(cycle_info['cycle_probs'])
                
                # Get recency info if timestamp provided
                recency_info = {}
                if current_time and hasattr(self, 'last_update_matrices'):
                    cycle_path = cycle_info['cycle_path']
                    edge_times = []
                    for i in range(len(cycle_path) - 1):
                        from_player, to_player = cycle_path[i], cycle_path[i + 1]
                        if from_player > to_player:
                            from_player, to_player = to_player, from_player
                        last_time = self.last_update_matrices[surface].get((from_player, to_player), 0)
                        edge_times.append(current_time - last_time if last_time > 0 else float('inf'))
                    
                    recency_info = {
                        'days_since_last_update': [t / (24 * 60 * 60) for t in edge_times],
                        'most_recent_days': min(edge_times) / (24 * 60 * 60) if edge_times else float('inf'),
                        'oldest_days': max(edge_times) / (24 * 60 * 60) if edge_times else float('inf'),
                    }
                
                cycle_results.append({
                    'match_idx': match_idx,
                    'player_a_idx': a,
                    'player_b_idx': b,
                    'bet_on_player_idx': cycle_info['bet_on_player_idx'],
                    'cycle_path': cycle_info['cycle_path'],
                    'min_cycle_prob': min_prob,
                    'avg_cycle_prob': avg_prob,
                    'cycle_strength': min_prob,  # Use min as strength measure
                    **recency_info
                })
        
        return cycle_results
    


    def calculate_match_local_intransitivity(self, surface: str, query_edges: torch.Tensor, 
                                           k_hops: int = 1, min_common: int = 1) -> List[float]:
        """
        Calculate local intransitivity measures for specific matches on a given surface.
        For each match, finds common opponents and calculates I(A) on the local subgraph.
        
        Args:
            surface: Surface to analyze ('Hard', 'Clay', 'Grass')
            query_edges: Tensor of shape (n_matches, 2) containing match pairs
            k_hops: Number of hops for neighborhood (1 for immediate triads)
            min_common: Minimum number of common opponents needed for calculation
            
        Returns:
            List of local intransitivity measures, one per match
        """
        if surface not in self.interaction_matrices:
            return [1.0] * len(query_edges)
        
        # Convert tensor to numpy array
        query_edges_np = query_edges.cpu().numpy().astype(int)
        
        # Use the vectorized local intransitivity calculator
        local_ia_values = calculate_match_local_intransitivity_vectorized(
            self.interaction_matrices,
            surface,
            query_edges_np,
            k_hops=k_hops,
            min_common=min_common
        )
        
        return local_ia_values

    def calculate_match_local_intransitivity_extended(self, surface: str, query_edges: torch.Tensor, 
                                                    k_hops: int = 1, min_common: int = 1) -> Dict[str, List[float]]:
        """
        Calculate extended local intransitivity measures for specific matches on a given surface.
        Returns all normalized measures (density-based) for each match.
        
        Args:
            surface: Surface to analyze ('Hard', 'Clay', 'Grass')
            query_edges: Tensor of shape (n_matches, 2) containing match pairs
            k_hops: Number of hops for neighborhood (1 for immediate triads)
            min_common: Minimum number of common opponents needed for calculation
            
        Returns:
            Dictionary containing lists of measures for each match:
            - 'intransitivity_measure': Basic I(A) values
            - 'density': Graph density values
            - 'total_norm': Total norm values
        """
        if surface not in self.interaction_matrices:
            empty_result = {
                'intransitivity_measure': [1.0] * len(query_edges),
                'density': [0.0] * len(query_edges),
                'total_norm': [0.0] * len(query_edges)
            }
            return empty_result
        
        # Convert tensor to numpy array
        query_edges_np = query_edges.cpu().numpy().astype(int)
        
        # Use the extended vectorized local intransitivity calculator
        extended_results = calculate_match_local_intransitivity_vectorized_extended(
            self.interaction_matrices,
            surface,
            query_edges_np,
            k_hops=k_hops,
            min_common=min_common
        )
        
        return extended_results

    def calculate_intransitivity_measure(self, surface_weights: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, Dict]]:
        """
        Calculate the I(A) intransitivity measure across all surfaces.
        
        Args:
            surface_weights: Optional weights for each surface (defaults to equal weighting)
            
        Returns:
            Dictionary containing:
            - overall_intransitivity: Weighted average I(A) across surfaces
            - surface_measures: Per-surface intransitivity breakdowns  
            - weighted_components: Weighted transitive/cyclic components
            - interpretation: Overall interpretation (predominantly_transitive/effectively_intransitive/balanced)
        """
        # Debug: Check matrix states
        matrix_info = {}
        total_nonzero = 0
        for surface, matrix in self.interaction_matrices.items():
            nnz = matrix.nnz if hasattr(matrix, 'nnz') else 'unknown'
            matrix_info[surface] = nnz
            if hasattr(matrix, 'nnz'):
                total_nonzero += matrix.nnz
        
        print(f"DEBUG: Matrix non-zero counts: {matrix_info}, Total: {total_nonzero}")
        
        if total_nonzero == 0:
            print("WARNING: All interaction matrices are empty! Cannot calculate intransitivity.")
            # Return default values instead of empty
            return {
                'overall_intransitivity': 1.0,
                'surface_measures': {},
                'weighted_components': {'transitive': 0.0, 'cyclic': 0.0},
                'interpretation': 'balanced'
            }
        
        # Use the calculate_surface_intransitivity function from model.graph.ia_compute
        results = calculate_surface_intransitivity(self.interaction_matrices, surface_weights)
        
        # Debug: Check results
        print(f"DEBUG: Intransitivity results: overall={results.get('overall_intransitivity', 'missing')}")
        
        # Add overall interpretation
        overall_ia = results['overall_intransitivity']
        if overall_ia < 1.0:
            interpretation = "predominantly_transitive"
        elif overall_ia > 1.0:
            interpretation = "effectively_intransitive"
        else:
            interpretation = "balanced"
        
        results['interpretation'] = interpretation
        return results

    def get_edge_features(self, surface, query_edges):
        """Per-match edge features from current PIM state.

        Returns (n, 9) tensor: [posterior, log_conf, staleness_yrs,
            a_serve, b_serve, a_return, b_return, a_bp, b_bp]
        """
        n = query_edges.size(0)
        feats = torch.zeros(n, 9)  # CPU; moved to device at return
        if surface not in self.interaction_matrices:
            return feats
        im = self.interaction_matrices[surface]
        cw = self.cumulative_weight[surface]
        lu = self.last_update_matrices[surface]
        h2h = self.h2h_stats  # Cross-surface running averages
        now = max((lu[k[0], k[1]] for k in lu.keys() if lu[k[0], k[1]] > 0), default=0)

        for idx, (a, b) in enumerate(query_edges.cpu().numpy().astype(int)):
            # Signed posterior: positive if A dominates B
            w_ab, w_ba = float(im.get((a, b), 0.0)), float(im.get((b, a), 0.0))
            feats[idx, 0] = w_ab if w_ab > 0 else (-w_ba if w_ba > 0 else 0.0)
            # Log confidence
            key = (min(a, b), max(a, b))
            feats[idx, 1] = float(np.log1p(cw.get(key, 0.0)))
            # Staleness (years since last H2H update; 10 = no record)
            last = float(lu.get(key, 0))
            feats[idx, 2] = float((now - last) / (365 * 24 * 60 * 60)) if last > 0 and now > 0 else 10.0
            # H2H stats: running averages (0.5 default = uninformative prior)
            # Check both orderings; swap A/B stats if pair is reversed
            if (a, b) in h2h:
                entry, swapped = h2h[(a, b)], False
            elif (b, a) in h2h:
                entry, swapped = h2h[(b, a)], True
            else:
                entry = None
            if entry is not None:
                order = ['b_serve', 'a_serve', 'b_return', 'a_return', 'b_bp', 'a_bp'] if swapped \
                    else ['a_serve', 'b_serve', 'a_return', 'b_return', 'a_bp', 'b_bp']
                for j, k in enumerate(order):
                    s, w = entry[k]
                    feats[idx, 3 + j] = float(s / w) if w > 1e-7 else 0.5
            else:
                feats[idx, 3:9] = 0.5
        return feats.to(device)

    def get_match_confidence_values(self, surface: str, query_edges: torch.Tensor) -> List[float]:
        """
        Extract cumulative weight (confidence) values c_ij for each match.

        Args:
            surface: Surface to analyze
            query_edges: Tensor of shape (n_matches, 2) containing match pairs

        Returns:
            List of confidence values (cumulative weights) for each matchup
        """
        if surface not in self.cumulative_weight:
            return [0.0] * len(query_edges)

        query_edges_np = query_edges.cpu().numpy().astype(int)
        confidence_values = []

        im = self.interaction_matrices[surface]
        for player_a, player_b in query_edges_np:
            # Ensure ordered pair
            a, b = (player_a, player_b) if player_a < player_b else (player_b, player_a)

            # Only report confidence if edge exists in (possibly pruned) graph
            has_edge = im.get((a, b), 0.0) != 0 or im.get((b, a), 0.0) != 0
            confidence = self.cumulative_weight[surface].get((a, b), 0.0) if has_edge else 0.0
            confidence_values.append(float(confidence))

        return confidence_values

    @staticmethod
    def _create_node_feature_matrix(gender="men"):
        """Create node feature matrix for all players"""
        # Load player data
        player_data = load_player_data(gender)
        
        # Get maximum index to determine matrix size
        max_idx = max(data['index'] for data in player_data.values())
        n_nodes = max_idx + 1
        
        # Initialize feature matrix (4 player features)
        feature_matrix = torch.zeros((n_nodes, 4))
        
        # Fill in features for each player
        for player_info in player_data.values():
            idx = player_info['index']
            # Set player features
            feature_matrix[idx] = torch.tensor(player_info['feature_vector'])
        
        return feature_matrix

