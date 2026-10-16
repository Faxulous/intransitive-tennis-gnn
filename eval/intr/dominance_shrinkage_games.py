#!/usr/bin/env python3
"""Shrinkage sensitivity of the GAME-PROPORTION dominance score.

The model's directed dominance edge magnitude is built from the proportion of GAMES
won, centred at 0.5 (player_interaction.py: centered_dominance = game_pct - 0.5),
accumulated across a pair's meetings. This is the quantity to stress-test for sparse
evidence - NOT a binary match-win share.

For each directed pair (u,v) we form the aggregate game-share
    D_uv = (games u won vs v) / (total games u-v played)
and its Beta(alpha,alpha) posterior mean
    D_shrunk = (games_u + alpha) / (total_games + 2*alpha)
which pulls D toward 0.5. We report, overall and per surface:
  - N edges, median meetings, median total games per edge,
  - Kendall tau between empirical and shrunken D (rank stability),
  - mean / median / 99th |D - D_shrunk| (magnitude shift).
alpha swept over {1,3,5} to show robustness to prior strength.

Run: python eval/intr/dominance_shrinkage_games.py
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from rich.console import Console
from rich.table import Table

con = Console(width=150)
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
os.chdir(ROOT); sys.path.insert(0, ROOT)
from model.graph.windowstacks import split_df_by_surface

ATP, WTA = 'data/atp_cleaned.csv', 'data/wta_cleaned.csv'
ALPHAS = [1, 3, 5]
SCOPE = ['graph', 'train', 'val']   # dominance graph entering test (excludes held-out test); see appendix
OUT_CSV = 'outputs/dominance_shrinkage_games.csv'


def directed_game_counts(df):
    """Per directed (winner-perspective) pair: sum of games each player won against the other.
    Uses A_games/B_games and the A/B identity (not winner/loser) so game tallies are exact."""
    a, b = df['Aname'].values, df['Bname'].values
    ag, bg = df['A_games'].values, df['B_games'].values
    rec = {}
    for i in range(len(df)):
        if not np.isfinite(ag[i]) or not np.isfinite(bg[i]):
            continue
        # canonical ordered pair (x<y); store games won by x and by y
        x, y, gx, gy = (a[i], b[i], ag[i], bg[i]) if a[i] < b[i] else (b[i], a[i], bg[i], ag[i])
        if (x, y) not in rec:
            rec[(x, y)] = [0.0, 0.0, 0]      # games_x, games_y, meetings
        rec[(x, y)][0] += gx; rec[(x, y)][1] += gy; rec[(x, y)][2] += 1
    rows = [(x, y, gx, gy, m) for (x, y), (gx, gy, m) in rec.items()]
    return pd.DataFrame(rows, columns=['x', 'y', 'gx', 'gy', 'meetings'])


def shrinkage_row(pairs, label, alpha):
    tot = pairs['gx'] + pairs['gy']
    pairs = pairs[tot > 0].copy(); tot = tot[tot > 0]
    D = pairs['gx'] / tot                                   # empirical game-share x over y
    D_s = (pairs['gx'] + alpha) / (tot + 2 * alpha)         # Beta(alpha,alpha) posterior mean
    delta = (D - D_s).abs()
    tau, p = kendalltau(D, D_s)
    return dict(slice=label, alpha=alpha, n_edges=len(pairs),
                med_meet=int(pairs['meetings'].median()), med_games=int(tot.median()),
                tau=round(tau, 3), tau_p=p,
                mean_delta=round(delta.mean(), 4), med_delta=round(delta.median(), 4),
                p99_delta=round(delta.quantile(0.99), 4))


def load_dev(path):  # per-gender chronological split, keep graph+train+val (drop held-out test)
    df = pd.read_csv(path, parse_dates=['Date'])
    sp = split_df_by_surface(df, 0.5, 0.1, 0.1)
    return pd.concat([sp[s][st] for s in sp for st in SCOPE], ignore_index=True)


def main():
    con.print("Load source data (graph+train+val; test held out)")
    src = pd.concat([load_dev(ATP), load_dev(WTA)], ignore_index=True)
    con.print(f"  {len(src):,} matches; surfaces {sorted(src['Surface'].dropna().unique())}\n")

    rows = []
    for alpha in ALPHAS:
        rows.append(shrinkage_row(directed_game_counts(src), 'Overall', alpha))
        for s in ['Clay', 'Grass', 'Hard']:
            rows.append(shrinkage_row(directed_game_counts(src[src['Surface'] == s]), s, alpha))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    for alpha in ALPHAS:
        tbl = Table(title=f"Game-share dominance: empirical D vs Beta({alpha},{alpha})-shrunken D")
        for c in ['Slice', 'N edges', 'Med meetings', 'Med games', 'Kendall tau', 'Mean |dD|', 'Med |dD|', '99th |dD|']:
            tbl.add_column(c, justify='right')
        for r in df[df.alpha == alpha].itertuples():
            tbl.add_row(r.slice, f'{r.n_edges:,}', str(r.med_meet), str(r.med_games),
                        f'{r.tau:.3f}', f'{r.mean_delta:.4f}', f'{r.med_delta:.4f}', f'{r.p99_delta:.4f}')
        con.print(tbl)
    con.print(f"\n  saved -> {OUT_CSV}")
    con.print("[dim]D = games won by x / total games (x,y); shrinks toward 0.5. Edge magnitude in the model "
              "is this game-share centred at 0.5, so this is the quantity of interest.[/dim]")


if __name__ == '__main__':
    main()
