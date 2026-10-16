#!/usr/bin/env python3
"""Shrinkage of the ASSEMBLED dominance score D^s_n(u,v) (direct version).

The games-share check (dominance_shrinkage_games.py) shrinks the within-match
evidence. This script shrinks the model's own assembled, time/surface/tier-weighted
posterior directly - the literal D^s_n(u,v) of the paper.

Per directed pair (u,v) on surface s the model stores
    D = weighted_sum / cumulative_weight            (posterior in [-0.5, 0.5])
We shrink toward 0 (neutral) by adding 2*alpha pseudo-observations of zero evidence:
    D_shrunk = weighted_sum / (cumulative_weight + 2*alpha)
Built on the dominance graph entering the held-out test period (graph+train+val, test
excluded) at the final T170 config, pooled across both genders and all three surfaces. We report:
  - N edges, fraction with effective weight < 1.5 (~ one meeting),
  - Kendall tau between |D| and |D_shrunk| (magnitude-ranking stability),
  - mean |D - D_shrunk| (magnitude shift),
  - sign flips (direction reversals) - provably zero, reported as a check.
alpha swept over {1,3,5}.

Run: python eval/intr/dominance_shrinkage_assembled.py
"""
import os, sys, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.graph.gnn import GraphEngine
from model.graph.windowstacks import split_df_by_surface, create_meta_stacks

con = Console(width=150)

CFG = 'tunes/tune_degree_postab_best.json'
ALPHAS = [1, 3, 5]
WEIGHT_1MEET = 1.5          # cumulative_weight below this ~= a single (decayed) meeting
OUT_CSV = 'outputs/dominance_shrinkage_assembled.csv'


def build_kw(p):  # mirror build_preds_w_ia.build_kw (graph-construction params only)
    surfaces = {"Hard":  {"Hard": 1.0, "Clay": p['HC'], "Grass": p['HG']},
                "Clay":  {"Hard": p['CH'], "Clay": 1.0, "Grass": p['CG']},
                "Grass": {"Hard": p['GH'], "Clay": p['GC'], "Grass": 1.0}}
    tq = {'GS': 1.0, 'Finals': p['beta_Finals'], '1000': p['beta_1000'], '500': p['beta_500']}
    return {'lr': p['lr'], 'K': p['K'], 'hidden': p['hidden'], 'layers': p['layers'],
            'activation': p['activation'], 'q': p['q'],
            'granularity': p['granularity'], 'lambda_param': p['lambda_param'],
            'max_edge_age': p['max_edge_age'], 'min_dom_pctile': p['min_dom_pctile'],
            'surfaces': surfaces, 'tournament_quality': tq,
            'mlp_head': False, 'n_edge_features': 0}


def collect_edges(gender, kw):  # build full graph, return per-edge (ws, cw)
    prefix = 'atp' if gender == 'men' else 'wta'
    df = pd.read_csv(f'data/{prefix}_cleaned.csv')
    ge = GraphEngine(**kw, gender=gender)
    splits = split_df_by_surface(df, 0.5, 0.1, 0.1)
    meta = create_meta_stacks(splits, gender)
    ge.pim.update_pim_from_stacks(meta, ['graph', 'train', 'val'])  # graph entering test (test held out)
    ws_all, cw_all = [], []
    for s in ge.pim.surfaces:
        cw = ge.pim.cumulative_weight[s].tocoo()
        ws = ge.pim.weighted_sum[s].todok()
        for i, j, w in zip(cw.row, cw.col, cw.data):
            if i < j and w > 1e-6:                 # upper triangle, real edges only
                ws_all.append(ws.get((i, j), 0.0)); cw_all.append(w)
    con.print(f"  {gender}: {len(cw_all):,} directed edges across 3 surfaces")
    return np.array(ws_all), np.array(cw_all)


def main():
    con.print("Build accumulated graph at T170")
    p = json.load(open(CFG))['params']
    kw = build_kw(p)
    ws, cw = [], []
    for g in ['men', 'women']:
        w, c = collect_edges(g, kw)
        ws.append(w); cw.append(c)
    ws = np.concatenate(ws); cw = np.concatenate(cw)
    D = ws / cw                                     # assembled posterior (signed)
    frac_1meet = (cw < WEIGHT_1MEET).mean()
    con.print(f"\n  {len(D):,} edges pooled; median cum-weight {np.median(cw):.2f}; "
              f"{frac_1meet:.1%} with weight < {WEIGHT_1MEET}\n")

    rows = []
    for a in ALPHAS:
        D_s = ws / (cw + 2 * a)                     # add 2a zero-evidence pseudo-obs
        delta = np.abs(D - D_s)
        tau, _ = kendalltau(np.abs(D), np.abs(D_s))
        flips = int(np.sum(np.sign(D) != np.sign(D_s)))
        rows.append(dict(alpha=a, n_edges=len(D), frac_1meeting=round(frac_1meet, 3),
                         tau_magnitude=round(tau, 3), mean_shift=round(delta.mean(), 4),
                         med_shift=round(np.median(delta), 4),
                         p99_shift=round(np.quantile(delta, 0.99), 4), sign_flips=flips))
    out = pd.DataFrame(rows); out.to_csv(OUT_CSV, index=False)

    tbl = Table(title="Assembled D^s_n(u,v): posterior vs Beta-shrunken (toward 0)")
    for c in ['alpha', 'N edges', 'Frac 1-meeting', 'tau |D|', 'Mean shift', 'Med shift', '99th shift', 'Sign flips']:
        tbl.add_column(c, justify='right')
    for r in out.itertuples():
        tbl.add_row(str(r.alpha), f'{r.n_edges:,}', f'{r.frac_1meeting:.1%}', f'{r.tau_magnitude:.3f}',
                    f'{r.mean_shift:.4f}', f'{r.med_shift:.4f}', f'{r.p99_shift:.4f}', str(r.sign_flips))
    con.print(tbl)
    con.print(f"\n  saved -> {OUT_CSV}")
    con.print("[dim]D = weighted_sum/cumulative_weight; shrink adds 2a zero pseudo-obs. Sign flips = 0 by "
              "construction (denominator always positive, numerator sign preserved).[/dim]")


if __name__ == '__main__':
    main()
