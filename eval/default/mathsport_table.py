#!/usr/bin/env python3
"""Appendix table extended_performance: earlier (MathSport) model on its original and extended test periods.

MS = preds_original.csv (the published conference predictions, 31 Oct 2023 - 8 Sep 2024).
E  = MS plus preds_extension.csv (the 844 later matches to 8 Jun 2025, regenerated; see README).

Run: python eval/default/mathsport_table.py
"""
import os
import pandas as pd
from rich.console import Console
from rich.table import Table

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MS_PREDS  = 'data/predictions/mathsport/preds_original.csv'
EXT_PREDS = 'data/predictions/mathsport/preds_extension.csv'
SURFACES  = ['Clay', 'Grass', 'Hard']


def score(df):  # (count, accuracy, Brier)
    return len(df), ((df['model_prob'] > 0.5) == df['Awin']).mean(), ((df['model_prob'] - df['Awin']) ** 2).mean()


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    ms = pd.read_csv(MS_PREDS)
    e = pd.concat([ms, pd.read_csv(EXT_PREDS)], ignore_index=True)

    tab = Table(title='Earlier model: original (MS) vs extended (E) test period')
    for col in ['Dataset', 'Count MS', 'Count E', 'Acc MS', 'Acc E', 'Brier MS', 'Brier E']:
        tab.add_column(col, justify='right')
    for name in SURFACES + ['All']:
        (n1, a1, b1), (n2, a2, b2) = (score(d if name == 'All' else d[d['Surface'] == name]) for d in (ms, e))
        tab.add_row(name, f'{n1:,}', f'{n2:,}', f'{a1:.3f}', f'{a2:.3f}', f'{b1:.3f}', f'{b2:.3f}')
    con = Console()
    con.print(tab)
    x = pd.read_csv(EXT_PREDS, parse_dates=['Date'])
    con.print(f"Original (MS) matches: {pd.to_datetime(ms['Date']).min():%d %b %Y} to {pd.to_datetime(ms['Date']).max():%d %b %Y}; "
              f"extension: {len(x):,} matches, {x['Date'].min():%d %b %Y} to {x['Date'].max():%d %b %Y}")


if __name__ == '__main__':
    main()
