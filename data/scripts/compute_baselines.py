#!/usr/bin/env python3
"""
Append Elo + BT baseline columns directly to cleaned CSV files.
Run once after loader.py builds the CSVs. Takes ~2-3 min.
"""
import sys, os, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from eval.baseline_models import calculate_elo_predictions, calc_bradley_terry_predictions
from utils.helpers import add_player_match_counts

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILES = [
    "data/atp_cleaned.csv",
    "data/wta_cleaned.csv",
    "data/atp_cleaned_VAL.csv",
    "data/wta_cleaned_VAL.csv",
]

BASELINE_COLS = [
    'A_match_count', 'B_match_count', 'match_count_diff',
    'A_experience', 'B_experience', 'match_experience_diff',
    'elo_surf_prob', 'elo_all_prob', 'elo_weighted_prob',
    'A_elo_surf', 'B_elo_surf', 'A_elo_all', 'B_elo_all',
    'A_elo_weighted', 'B_elo_weighted', 'bt_prob',
]


def process_file(path):
    """Read CSV, compute baselines, write back with new columns."""
    path = Path(path)
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return

    df = pd.read_csv(path, low_memory=False)

    # Drop existing baseline cols if re-running
    existing = [c for c in BASELINE_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"  {path.name}: dropped {len(existing)} existing baseline cols")

    print(f"  {path.name}: {len(df):,} matches — computing match counts...")
    df = add_player_match_counts(df)

    print(f"  {path.name}: computing Elo...")
    df = calculate_elo_predictions(df)

    print(f"  {path.name}: computing Bradley-Terry...")
    df = calc_bradley_terry_predictions(df)

    df.to_csv(path, index=False)
    print(f"  {path.name}: saved with {len(BASELINE_COLS)} baseline columns")
    return len(df)


if __name__ == "__main__":
    t0 = time.time()
    print("APPENDING BASELINES TO CLEANED CSVs")

    # Process full datasets in parallel, then VAL datasets
    for pair in [FILES[:2], FILES[2:]]:
        existing = [f for f in pair if Path(f).exists()]
        if not existing:
            continue
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(process_file, existing))

    print(f"\nDone in {time.time()-t0:.0f}s")
