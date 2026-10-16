#!/usr/bin/env python3
"""H2H sample-size distribution.

Two distributions: (1) pair-level - meeting counts across all unique unordered
(u,v) pairs in the full source; (2) match-level - prior H2H counts at the time
each test-set prediction is made. Quantifies how sparse the H2H evidence
behind the dominance estimates is.
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table

con = Console(width=120)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATP        = 'data/atp_cleaned.csv'
WTA        = 'data/wta_cleaned.csv'
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
OUT_CSV    = 'outputs/h2h_dist.csv'


def main():
    src = pd.concat([
        pd.read_csv(ATP, parse_dates=['Date'], usecols=['Date','Aname','Bname']),
        pd.read_csv(WTA, parse_dates=['Date'], usecols=['Date','Aname','Bname']),
    ], ignore_index=True).sort_values('Date').reset_index(drop=True)
    src['pair'] = [tuple(sorted([a, b])) for a, b in zip(src['Aname'], src['Bname'])]
    src['n_prior'] = src.groupby('pair').cumcount()                 # 0 on first ever meeting
    pair_counts = src.groupby('pair').size()
    con.print(f"{len(src):,} matches; {len(pair_counts):,} unique pairs")

    # ── Pair-level distribution ────────────────────────────────────────────────
    rows_pair = [
        ("N pairs",                  f"{len(pair_counts):,}"),
        ("Mean meetings/pair",       f"{pair_counts.mean():.2f}"),
        ("Median",                   f"{pair_counts.median():.0f}"),
        ("Max",                      f"{pair_counts.max():,}"),
        ("Pairs with >=2 meetings",  f"{(pair_counts>=2).sum():,} ({(pair_counts>=2).mean()*100:.1f}%)"),
        ("Pairs with >=3 meetings",  f"{(pair_counts>=3).sum():,} ({(pair_counts>=3).mean()*100:.1f}%)"),
        ("Pairs with >=5 meetings",  f"{(pair_counts>=5).sum():,} ({(pair_counts>=5).mean()*100:.1f}%)"),
        ("Pairs with >=10 meetings", f"{(pair_counts>=10).sum():,} ({(pair_counts>=10).mean()*100:.1f}%)"),
    ]
    t1 = Table(title="Pair level: meetings per unique (u,v)")
    t1.add_column("Stat", justify="right"); t1.add_column("Value", justify="right")
    for k, v in rows_pair: t1.add_row(k, v)
    con.print(t1)

    # ── Match-level distribution: prior H2H count at prediction time ──────────
    test = pd.read_csv(TEST_PREDS, parse_dates=['Date'], usecols=['Date','Aname','Bname'])
    test['pair'] = [tuple(sorted([a, b])) for a, b in zip(test['Aname'], test['Bname'])]
    merged = test.merge(src[['Date','pair','n_prior']], on=['Date','pair'], how='left')
    # Multiple source rows for same (Date,pair) - take min (= the row that *was* the test match)
    merged = merged.groupby(['Date','Aname','Bname'], as_index=False)['n_prior'].min()
    if merged['n_prior'].isna().any():
        raise ValueError(f"{merged['n_prior'].isna().sum()} test matches did not merge into source")
    n = merged['n_prior'].astype(int)

    bins = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,10**9)]
    labs = ["0", "1", "2", "3", "4", "5+"]
    t2 = Table(title=f"Table 7: pairs by total meetings ({len(pair_counts):,} pairs) and test matches by prior meetings "
                     f"({len(n):,}; mean {n.mean():.2f}, median {n.median():.0f}, max {n.max()})")
    for c in ["Meetings", "Pairs", "Pairs %", "Test matches", "Test %", "Test cum %"]:
        t2.add_column(c, justify="right")
    cum = 0
    rows_match = []
    for (lo, hi), lab in zip(bins, labs):
        m = int(((n >= lo) & (n <= hi)).sum()); cum += m
        k = int(pair_counts.between(lo, hi).sum())  # pairs exist only once they have met: no zero row
        t2.add_row(lab, f"{k:,}" if lo else '-', f"{k/len(pair_counts)*100:.1f}" if lo else '-',
                   f"{m:,}", f"{m/len(n)*100:.1f}", f"{cum/len(n)*100:.1f}")
        rows_match.append((lab, m, m/len(n)*100))
    con.print(t2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    pair_dist = pair_counts.value_counts().sort_index().rename_axis('meetings').reset_index(name='n_pairs').assign(level='pair')
    match_dist = pd.DataFrame(rows_match, columns=['bin','n_matches','pct']).assign(level='match')
    pd.concat([pair_dist.rename(columns={'meetings':'bin'}).astype({'bin':'str'}), match_dist],
              ignore_index=True).to_csv(OUT_CSV, index=False)
    con.print(f"saved -> {OUT_CSV}")


if __name__ == '__main__':
    main()
