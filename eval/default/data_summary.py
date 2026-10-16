#!/usr/bin/env python3
"""Section 3 dataset numbers: date span, matches and players per tour, and the validation/test splits.

Matches and players come from the cleaned match files (data/{atp,wta}_cleaned.csv, built by
data/scripts/loader.py); the splits from the prediction files of each set.

Run: python eval/default/data_summary.py
"""
import os
import pandas as pd
from rich.console import Console
from rich.table import Table

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCHES = {'men': 'data/atp_cleaned.csv', 'women': 'data/wta_cleaned.csv'}
SPLITS  = {'Validation': 'data/predictions/preds_w_ia_val.csv', 'Test': 'data/predictions/preds_w_ia.csv'}


def main():
    con, d = Console(width=120), {g: pd.read_csv(f, parse_dates=['Date']) for g, f in MATCHES.items()}
    tab = Table(title="Section 3: matches and players per tour")
    for c in ['Tour', 'Matches', 'Players', 'First match', 'Last match', 'Tiers']:
        tab.add_column(c, justify='right')
    for g, x in [*d.items(), ('both', pd.concat(d.values()))]:
        tiers = ', '.join(f'{k} {v:,}' for k, v in x['tourney_level'].value_counts().items())
        tab.add_row(g, f'{len(x):,}', f"{pd.concat([x.Aname, x.Bname]).nunique():,}" if g != 'both' else '',
                    f'{x.Date.min():%d %b %Y}', f'{x.Date.max():%d %b %Y}', tiers, end_section=g == 'women')
    con.print(tab)
    tab = Table(title="Section 3: chronological validation and test sets")
    for c in ['Set', 'Tour', 'Matches', 'From', 'To']:
        tab.add_column(c, justify='right')
    for s, f in SPLITS.items():
        p = pd.read_csv(f, parse_dates=['Date'], usecols=['Date', 'gender_source'])
        for k, (g, x) in enumerate([*p.groupby('gender_source'), ('both', p)]):
            tab.add_row(s if k == 0 else '', g, f'{len(x):,}', f'{x.Date.min():%d %b %Y}', f'{x.Date.max():%d %b %Y}',
                        end_section=g == 'both')
    con.print(tab)


if __name__ == '__main__':
    main()
