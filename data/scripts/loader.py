#!/usr/bin/env python3
"""
Build cleaned match datasets from tennis-data.co.uk spreadsheets (downloaded by hand, see README).

Pipeline: yearly xls(x) -> tier filter -> drop walkovers/no-odds/no-sets
-> seeded A/B shuffle -> Shin implied probs -> windows -> write
data/{atp,wta}_cleaned{,_VAL}.csv + refresh data/player_data/{gender}.csv.

Run: python data/scripts/loader.py
"""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import shin
from pathlib import Path
from rich.console import Console

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8                                    # A/B shuffle reproducibility
YEARS = range(2012, 2026)
DATE_START, DATE_END = '2012-01-01', '2025-10-16'
RAW_DIR = Path('data/raw_tennisdata')       # {atp,wta}/{year}.xls(x), as downloaded
MEN_TIERS = ['Grand Slam', 'Masters 1000', 'ATP500', 'Masters Cup']
WOMEN_TIERS = ['Grand Slam', 'WTA1000', 'WTA500', 'Tour Championships']
DROP_COMMENTS = {'Walkover', 'Walkoer', 'Sched', 'Cancelled'}   # never played (incl. typo)
KEEP_COMMENTS = {'Completed', 'Retired', 'Rrtired', 'Awarded', 'Disqualified'}
ROUND_ORDER = ['Round Robin', '1st Round', '2nd Round', '3rd Round', '4th Round',
               'Quarterfinals', 'Semifinals', 'The Final']
SERVE_STAT_COLS = [f'{s}_{c}' for c in ['aces', 'df', '1st_in', '1st_total', '1st_won',
                   '1st_won_total', '2nd_won', '2nd_won_total', 'return_won', 'return_total',
                   'bp_won', 'bp_total', 'total_pts'] for s in ['A', 'B']]  # schema-required, NaN (source lacks them)

# tennis-data renames players mid-sample (convention break ~Apr 2016 + punctuation
# drift): minority alias -> canonical. Each merge verified against OnCourt records
# by (date, opponent) join. Same-surname DISTINCT players kept apart: Kuznetsov
# Al./An., Rodionova An./Ar., Pliskova Ka./Kr., Cerundolo F./J.M., Zahlavova S.
ALIAS = {
    # men
    'Bautista R.': 'Bautista Agut R.', 'Ramos A.': 'Ramos-Vinolas A.',
    'Carreno-Busta P.': 'Carreno Busta P.', 'Del Potro J. M.': 'Del Potro J.M.',
    'McDonald M.': 'Mcdonald M.', 'Herbert P.H': 'Herbert P.H.',
    'Herbert P.': 'Herbert P.H.', 'Herbert P-H.': 'Herbert P.H.',
    'Lu Y.': 'Lu Y.H.', 'Kuznetsov A.': 'Kuznetsov Al.',     # 2012 AO row = Alex
    "O'Connell C.": 'O Connell C.', 'Zhang Z.': 'Zhang Ze.', 'Zhang Ze': 'Zhang Ze.',
    'Galan D.': 'Galan D.E.', 'Tseng C. H.': 'Tseng C.H.', 'Varillas J. P.': 'Varillas J.P.',
    'Munoz De La Nava D.': 'Munoz-De La Nava D.', 'Viola Mat.': 'Viola M.',
    'McCabe J.': 'Mccabe J.', 'Blanch Dar.': 'Blanch D.',    # both rows = Darwin
    'Dolgopolov A.': 'Dolgopolov O.',
    # women
    'Collins D.R.': 'Collins D.', 'Fernandez L.': 'Fernandez L.A.',
    'Schmiedlova A.K.': 'Schmiedlova A.', 'Teichmann J.B.': 'Teichmann J.',
    'Zahlavova Strycova B.': 'Strycova B.', 'Soler-Espinosa S.': 'Soler Espinosa S.',
    'McNally C.': 'Mcnally C.', 'Duque-Marino M.': 'Duque Marino M.',
    'Badosa Gibert P.': 'Badosa P.', 'Badosa Gibert. P.': 'Badosa P.',
    'Xu Y-F.': 'Xu Y.F.', 'Lucic-Baroni M.': 'Lucic M.',
    'Saville D.': 'Gavrilova D.',                            # same player, married name
    'Kerkhove L.': 'Pattinama Kerkhove L.', 'Wang Y.': 'Wang Yaf.',
}
# 'Wang X.' is TWO players: Xinyu (2018 row) and Xiyu (2019 rows) - split by date
WANG_X_CUTOFF = '2019-01-01'

# Source typo rows to drop, (gender, date, winner, loser): Rome 2020 final has
# Halep's odds recorded as 52.0 (true ~1.4), inverting the implied probability
EXCLUDE = [('women', '2020-09-21', 'Halep S.', 'Pliskova Ka.')]

con = Console()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def raw_paths(gender):
    """One local spreadsheet per year; fail loudly if any is missing."""
    tour = 'atp' if gender == 'men' else 'wta'
    paths = [sorted((RAW_DIR / tour).glob(f'{year}.xls*')) for year in YEARS]
    missing = [y for y, p in zip(YEARS, paths) if len(p) != 1]
    if missing:
        raise FileNotFoundError(f'Need exactly one {RAW_DIR}/{tour}/{{year}}.xls(x) for {missing}: '
                                'download from http://www.tennis-data.co.uk/alldata.php (see README, Data)')
    return [p[0] for p in paths]


def read_raw(path):
    """Excel engine by magic bytes (server serves either format under either extension)."""
    eng = 'openpyxl' if path.read_bytes()[:2] == b'PK' else 'xlrd'
    return pd.read_excel(path, engine=eng)


def shin_implieds(odds_a, odds_b):
    """Shin implied probs + overround (z) for each odds pair."""
    res = [shin.calculate_implied_probabilities([a, b], full_output=True)
           for a, b in zip(odds_a, odds_b)]
    return (np.array([r['implied_probabilities'][0] for r in res]),
            np.array([r['implied_probabilities'][1] for r in res]),
            np.array([r['z'] for r in res]))


def load_gender(gender, tiers, date_start=DATE_START, date_end=DATE_END):
    """Read, filter, and format one tour into the cleaned-CSV schema."""
    df = pd.concat([read_raw(p) for p in raw_paths(gender)], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if gender == 'women':  # from ALL years, so VAL builds label Premier events identically
        t1000 = set(df.loc[df['Tier'] == 'WTA1000', 'Tournament'])
    df = df[(df['Date'] >= date_start) & (df['Date'] <= date_end)]
    n0 = len(df)

    # Tier filter. WTA pre-2021 'Premier' = 1000/500-equivalent (both kept);
    # split by whether the tournament carries the WTA1000 label post-rebrand.
    if gender == 'men':
        df['tourney_level'] = df['Series']
    else:
        df['tourney_level'] = np.where(df['Tier'] == 'Premier',
                                       np.where(df['Tournament'].isin(t1000), 'WTA1000', 'WTA500'),
                                       df['Tier'])
    df = df[df['tourney_level'].isin(tiers)]
    n1 = len(df)

    # Played-match filters: drop never-played, missing/invalid odds, <1 completed set,
    # and rows whose per-set games don't cover the completed sets
    unknown = set(df['Comment'].unique()) - KEEP_COMMENTS - DROP_COMMENTS
    if unknown:
        raise ValueError(f'Unknown Comment values: {unknown}')
    df = df[~df['Comment'].isin(DROP_COMMENTS)]
    n2 = len(df)
    df = df[df['PSW'].notna() & df['PSL'].notna() & (df['PSW'] >= 1.01) & (df['PSL'] >= 1.01)]
    n3 = len(df)
    n_sets = 5 if gender == 'men' else 3  # WTA files only carry W1..W3
    total_sets = df['Wsets'].fillna(0) + df['Lsets'].fillna(0)
    set_pairs = sum((df[f'W{i}'].notna() & df[f'L{i}'].notna()).astype(int) for i in range(1, n_sets + 1))
    df = df[(total_sets >= 1) & (set_pairs >= total_sets)]
    n4 = len(df)
    df = df.drop_duplicates(subset=['Date', 'Winner', 'Loser'], keep='first')

    bad_surface = set(df['Surface'].unique()) - {'Hard', 'Clay', 'Grass'}
    if bad_surface:
        raise ValueError(f'Unexpected surfaces: {bad_surface}')
    bad_rounds = set(df['Round'].unique()) - set(ROUND_ORDER)
    if bad_rounds:
        raise ValueError(f'Unexpected rounds: {bad_rounds}')

    for col in ['Winner', 'Loser']:
        df[col] = df[col].str.strip().replace(ALIAS)
        wx = df[col] == 'Wang X.'
        df.loc[wx, col] = np.where(df.loc[wx, 'Date'] < WANG_X_CUTOFF, 'Wang Xin.', 'Wang Xiy.')
    for g, date, w, l in EXCLUDE:
        if g == gender:
            typo = (df['Date'] == date) & (df['Winner'] == w) & (df['Loser'] == l)
            if typo.sum() != 1:
                raise ValueError(f'EXCLUDE row not found exactly once: {date} {w} d. {l}')
            df = df[~typo]

    # Source bugs: 2022 ATP Finals has blank Best of; 13 Wimbledon rows say bo3.
    # Men's Grand Slams are bo5 by rule, every other kept event is bo3.
    if gender == 'men':
        df.loc[df['tourney_level'] == 'Grand Slam', 'Best of'] = 5
    df.loc[df['Best of'].isna(), 'Best of'] = 3

    # Deterministic order, then seeded winner/loser -> A/B shuffle
    df['Round'] = pd.Categorical(df['Round'], categories=ROUND_ORDER, ordered=True)
    df = df.sort_values(['Date', 'Round', 'Tournament', 'Winner']).reset_index(drop=True)
    mask = np.random.default_rng(SEED).random(len(df)) < 0.5  # True = winner becomes B
    out = pd.DataFrame({'match_id': range(len(df)),
                        'tourney_name': df['Tournament'], 'Date': df['Date'],
                        'Round': df['Round'].astype(str), 'Surface': df['Surface'],
                        'tourney_level': df['tourney_level'],
                        'Aname': np.where(mask, df['Loser'], df['Winner']),
                        'Bname': np.where(mask, df['Winner'], df['Loser']),
                        'Asets': np.where(mask, df['Lsets'], df['Wsets']),
                        'Bsets': np.where(mask, df['Wsets'], df['Lsets']),
                        'Awin': (~mask).astype(int),
                        'PSA': np.where(mask, df['PSL'], df['PSW']),
                        'PSB': np.where(mask, df['PSW'], df['PSL']),
                        'RA': np.where(mask, df['LRank'], df['WRank']),
                        'RB': np.where(mask, df['WRank'], df['LRank']),
                        'Best of': df['Best of'].astype(int), 'gender': gender})
    for i in range(1, 6):
        out[f'A{i}'] = np.where(mask, df[f'L{i}'], df[f'W{i}']) if i <= n_sets else np.nan
        out[f'B{i}'] = np.where(mask, df[f'W{i}'], df[f'L{i}']) if i <= n_sets else np.nan
    out['A_games'] = out[[f'A{i}' for i in range(1, 6)]].sum(axis=1)  # incl. partial set before ret.
    out['B_games'] = out[[f'B{i}' for i in range(1, 6)]].sum(axis=1)
    out['A_game_pct'] = out['A_games'] / (out['A_games'] + out['B_games'])
    out['A_set_pct'] = out['Asets'] / (out['Asets'] + out['Bsets'])
    out['PSA_ip'], out['PSB_ip'], out['PS_overround'] = shin_implieds(out['PSA'], out['PSB'])
    for c in SERVE_STAT_COLS:
        out[c] = np.nan

    # Windows: Year + Tournament + Round, idx sequential in (Date, Round) order
    out['window'] = (out['Date'].dt.year.astype(str) + ' ' + out['tourney_name']
                     + ' ' + out['Round'])
    out['window_idx'] = pd.factorize(out['window'])[0]  # sequential in first-appearance order

    con.print(f"  {gender}: raw {n0} -> tier {n1} -> played {n2} -> odds {n3} "
              f"-> sets {n4} -> dedup {len(out)}; {out.window_idx.nunique()} windows, "
              f"{len(set(out.Aname) | set(out.Bname))} players")
    return out


def generate_player_data(df, gender, out_dir="data/player_data"):
    """Refresh player_data CSV: keep biometrics for known players, add new ones."""
    src = pd.read_csv(Path(out_dir) / f"{gender}.csv")
    match_names = sorted(set(df['Aname']).union(set(df['Bname'])))
    src_names = set(src['player_name'])

    missing = [n for n in match_names if n not in src_names]
    if missing:
        src = pd.concat([src, pd.DataFrame({'player_name': missing})], ignore_index=True)
        con.print(f"  {len(missing)} new players added: {missing[:5]}...")

    # Filter to only players in this dataset, reassign contiguous indices
    src = src[src['player_name'].isin(match_names)].copy()
    src = src.sort_values('player_name').reset_index(drop=True)
    src['index'] = range(len(src))

    out = Path(out_dir) / f"{gender}.csv"
    if 'height' not in src:  # public package: names only (tennisexplorer.com terms forbid publishing attributes)
        src[['player_name', 'index']].to_csv(out, index=False)
        con.print(f"  Saved {len(src)} players -> {out} (no attributes)")
        return src

    h_m = src['height'].notna() & src['weight'].notna()
    src.loc[h_m, 'bmi'] = src.loc[h_m, 'weight'] / (src.loc[h_m, 'height'] / 100) ** 2

    # Fill NaN biometrics with medians (same logic as load_player_data)
    for col in ['height', 'weight', 'bmi', 'righthanded']:
        median = src[col].median()
        if pd.notna(median):
            src[col] = src[col].fillna(median)
        else:
            src[col] = src[col].fillna(180.0 if col == 'height' else
                                       75.0 if col == 'weight' else
                                       23.1 if col == 'bmi' else 1.0)

    src.to_csv(out, index=False)
    n_real = sum(1 for n in match_names if n in src_names)
    con.print(f"  Saved {len(src)} players -> {out} ({n_real} with biometrics)")
    return src


def build_dataset(gender, tiers, date_end=DATE_END, save=True, suffix=''):
    """Full pipeline: load -> player data -> save."""
    con.print(f"\n[bold]BUILDING DATASET: {gender.upper()}{' (VAL)' if suffix else ''}[/bold]")
    df = load_gender(gender, tiers=tiers, date_end=date_end)
    con.print(f"  Surfaces: {dict(df['Surface'].value_counts())}")
    con.print(f"  Tiers: {dict(df['tourney_level'].value_counts())}")

    if not suffix:  # Only generate player_data for full builds (not _VAL)
        generate_player_data(df, gender)

    if save:
        prefix = 'atp' if gender == 'men' else 'wta'
        out = f"data/{prefix}_cleaned{suffix}.csv"
        df.to_csv(out, index=False)
        con.print(f"  Saved: {out} ({len(df):,} matches)")
    return df


def compute_val_end(df, g_pct=0.5, t_pct=0.1, v_pct=0.1):
    """Compute the date where WF's val split ends (= val file truncation point).
    WF splits: graph(50%) + train(10%) + val(10%) + test(30%). The boundary date
    itself is excluded so matches on it fall only in test, never duplicated into val."""
    cutoff_idx = int(len(df) * (g_pct + t_pct + v_pct))
    sorted_df = df.sort_values(['window_idx', 'match_id'])
    cutoff_date = sorted_df.iloc[cutoff_idx]['Date']
    return sorted_df.loc[sorted_df['Date'] < cutoff_date, 'Date'].max()  # strictly before cutoff


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    men = build_dataset('men', MEN_TIERS)
    women = build_dataset('women', WOMEN_TIERS)

    men_val_end = compute_val_end(men)
    women_val_end = compute_val_end(women)
    con.print(f"\nVal truncation: men={men_val_end.date()}, women={women_val_end.date()}")

    build_dataset('men', MEN_TIERS, date_end=str(men_val_end.date()), suffix='_VAL')
    build_dataset('women', WOMEN_TIERS, date_end=str(women_val_end.date()), suffix='_VAL')
    con.print("\n[bold green]Done.[/bold green] Next: compute_baselines.py, then build_preds_w_ia.py")
