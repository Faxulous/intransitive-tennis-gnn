#!/usr/bin/env python3
"""Two-part ablation for Section 4.5 (complexity justification).

Runs on VAL split — design-time ablation that informs model-component decisions
(which components to retain/drop before retuning). TEST is held out and not
touched here. Re-run on TEST only after the architecture is locked from VAL.

Part A — Stepwise cumulative M0->M5. Each row adds ONE component to its predecessor:
  M0 id-only/no-graph -> M1 +features/no-graph -> M2 +graph -> M3 +cross
  -> M4 +multihop -> M5 +bakes (TQ buckets + age decay)

Part B — Leave-one-out from M5_full. Each row removes ONE component from the full model.
  Grouped: Node features / Edge weights / Structural.

All GNN rows use T0-warm tuned numerics (tune_tqbins_best.json), varying only
the flag(s) specified per row. Gender-stratified Brier on VAL.

Run (toy): python eval/ablation.py --test
Run (full): python eval/ablation.py
Table 3 from the included results only: python eval/ablation.py --table
"""
import sys, os, json, time, argparse, warnings, torch
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.graph.player_interaction import PlayerInteractionMatrix
from model.graph.gnn import GraphEngine
from model.walk_forward_validation import walk_forward_validation, precompute_walk_forward_data
from model.seed import set_seed

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
LR_MULT = 0.5                            # T0-warm convention: half tuned lr for stability
RETRAIN_EVERY = 38                       # retrain every 38 snapshots (Section 4.3)
ZERO_COLS_TIDY = [0, 1, 2, 3, 8]         # bake: body (0-3) + bp_avg (8) dropped
ZERO_COLS_ALL  = list(range(14))         # all features zeroed (used for M0 + LOO contrasts)
ID_COLS = [0, 1, 2, 3]                   # M0 random-identity vector lives in static slots
JSON_PATH = 'tunes/tune_tqbins_best.json'
OUT_PATH = 'outputs/clean_ablation_val.csv'
DATA = {'men': 'data/atp_cleaned_VAL.csv', 'women': 'data/wta_cleaned_VAL.csv'}
TABLE3 = [('M0_id_nograph', 'Dominance graph (entire)'), ('loo_edge_age', 'Edge-age decay'),     # paper row order
          ('loo_struct_K1', 'Multi-hop receptive field (L=2)'), ('loo_node_degree', 'In/out-degree node features'),
          ('loo_edge_cross', 'Cross-surface edges'), ('loo_edge_uniform_w', 'Dominance edge weights'),
          ('loo_struct_q0', 'Directional weighting (q)'), ('add_node_static', 'Static player attributes (add)'),
          ('loo_edge_tq', 'Uniform tournament-tier weights'), ('loo_edge_cross_sym', 'Symmetric surface transfer'),
          ('loo_struct_gran_games', 'Games-level dominance signal')]
NOT_PUBLIC = {'add_node_static'}  # needs player attributes that cannot be published

# Feature groups (surviving cols after ZERO_COLS_TIDY bake).
# tennis-data rebuild: serve-stat cols 6-13 are identically zero (source has no
# serve statistics), so the old srv_ret/1st_serve/2nd_win/ace_df LOO arms are gone.
FEAT_GROUPS = {
    'degree':    [4, 5],      # in/out degree from graph
}

warnings.filterwarnings('ignore', category=FutureWarning)
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
con = Console(width=120)
_ORIG_CSG = PlayerInteractionMatrix.create_surface_graph   # original class method, restored after each ablation


def brier(y, p): return float(np.mean((p - y) ** 2))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GNN RUNNER — single config spec dict drives everything
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_ge(P, gender, nfm, cfg, id_features=None):
    """Build engine + a CLASS-level create_surface_graph patch (returns (ge, csg)).
       cfg overrides: K, layers, cross, tq, zero_cols, granularity, q,
       max_edge_age, uniform_edge_w (bool), no_graph (bool), use_id_nfm (bool)."""
    ge = GraphEngine(
        lr=P['lr']*LR_MULT,
        granularity=cfg.get('granularity', P['granularity']),
        K=cfg['K'], hidden=P['hidden'], layers=cfg['layers'],
        activation=P['activation'], gender=gender,
        surfaces=cfg['cross'], lambda_param=P['lambda_param'],
        tournament_quality=cfg['tq'],
        max_edge_age=cfg.get('max_edge_age', P['max_edge_age']),
        min_dom_pctile=P['min_dom_pctile'],
        node_feature_matrix=nfm,
        q=cfg.get('q', P['q']),
        mlp_head=False, n_edge_features=0,
    )
    zc, uni = cfg['zero_cols'], cfg.get('uniform_edge_w', False)
    no_graph = cfg.get('no_graph', False)
    use_id = cfg.get('use_id_nfm', False)
    # Patch the CLASS method (takes self) so the val_pim copy used for TRAINING is ablated too —
    # not just pred_pim. Instance-patching ge.pim left training on un-ablated features (a train/serve
    # mismatch) that spuriously inflated every feature/graph-removal row. Installed+restored in run_gnn.
    def csg(self, surface, current_time=None, _o=_ORIG_CSG, _c=zc, _u=uni):
        _o(self, surface, current_time)
        sg = self.surface_graphs[surface]
        for c in _c:
            sg['X_real'][:, c] = 0.0
        if _u:  # Collapse magnitudes to ±1 (preserve direction, kill weights)
            sg['edge_weight'] = torch.sign(sg['edge_weight'])
        if use_id and id_features is not None:  # Inject random per-player identity post-zero
            dev = sg['X_real'].device
            sg['X_real'][:, ID_COLS] = id_features.to(dev)
        if no_graph:  # Collapse to self-loops only -> MagNet acts as per-node MLP
            n = sg['X_real'].shape[0]
            dev = sg['edge_index'].device
            sg['edge_index'] = torch.stack([torch.arange(n, device=dev),
                                            torch.arange(n, device=dev)])
            sg['edge_weight'] = torch.ones(n, device=dev)
    return ge, csg


def run_gnn(P, gender, df, wf, nfm, cfg, epochs, id_features=None):
    set_seed(SEED); torch.cuda.empty_cache()
    ge, csg = build_ge(P, gender, nfm, cfg, id_features=id_features)
    init_ep, retrain_ep = epochs
    try:
        PlayerInteractionMatrix.create_surface_graph = csg   # train (val_pim copy) + predict both ablated
        pred = walk_forward_validation(
            ge, df, gender, seed=SEED, verbose=False,
            initial_epochs=init_ep, subsequent_epochs=retrain_ep,
            retrain_every=RETRAIN_EVERY, cached_data=wf,
        )
    finally:
        PlayerInteractionMatrix.create_surface_graph = _ORIG_CSG
    valid = pred['model_prob'].notna() & pred['Awin'].notna()
    y = pred.loc[valid, 'Awin'].values
    p = pred.loc[valid, 'model_prob'].values
    return brier(y, p), int(valid.sum())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='Toy run: 2000 rows, 5/2 epochs')
    ap.add_argument('--part', choices=['A','B','both'], default='both')
    ap.add_argument('--only', default=None, help='Substring filter on row names (e.g. gran_games)')
    ap.add_argument('--append', action='store_true', help='Append to existing CSV instead of overwrite')
    ap.add_argument('--table', action='store_true', help=f'Only print Table 3 from {OUT_PATH} (no training)')
    args = ap.parse_args()
    if args.table:
        return print_tables(pd.read_csv(OUT_PATH))

    EPOCHS = (5, 2) if args.test else (100, 30)
    DATA_LIMIT = 2000 if args.test else None

    with open(JSON_PATH) as f:
        P = json.load(f)['params']

    # Cross-surface variants
    NO_CROSS = {s: {s2: (1.0 if s == s2 else 0.0) for s2 in ['Hard','Clay','Grass']} for s in ['Hard','Clay','Grass']}
    FULL_CROSS = {
        "Hard":  {"Hard": 1.0,     "Clay": P['HC'], "Grass": P['HG']},
        "Clay":  {"Hard": P['CH'], "Clay": 1.0,     "Grass": P['CG']},
        "Grass": {"Hard": P['GH'], "Clay": P['GC'], "Grass": 1.0},
    }
    # Symmetric cross-surface: 3 params (avg each pair)
    s_HC, s_HG, s_CG = (P['HC']+P['CH'])/2, (P['HG']+P['GH'])/2, (P['CG']+P['GC'])/2
    SYM_CROSS = {
        "Hard":  {"Hard": 1.0, "Clay": s_HC, "Grass": s_HG},
        "Clay":  {"Hard": s_HC, "Clay": 1.0, "Grass": s_CG},
        "Grass": {"Hard": s_HG, "Clay": s_CG, "Grass": 1.0},
    }
    TQ_FLAT  = {'GS': 1.0, 'Finals': 1.0, '1000': 1.0, '500': 1.0}
    TQ_TIERS = {'GS': 1.0, 'Finals': P['beta_Finals'],
                '1000': P['beta_1000'], '500': P['beta_500']}

    # M5_full baseline config (shared across both parts) — tuned arch from JSON
    M5 = {'K': P['K'], 'layers': P['layers'], 'cross': FULL_CROSS, 'tq': TQ_TIERS,
          'zero_cols': ZERO_COLS_TIDY, 'max_edge_age': P['max_edge_age']}

    # Part A — stepwise cumulative (M0 -> M5)
    # Each step adds ONE component to its predecessor.
    BASE_NOGRAPH = {**M5, 'K':1, 'layers':1, 'cross':NO_CROSS, 'tq':TQ_FLAT,
                    'max_edge_age': 0}
    STEPS_A = [
        ('M0_id_nograph',  {**BASE_NOGRAPH, 'zero_cols': ZERO_COLS_ALL,
                            'no_graph': True, 'use_id_nfm': True}),
        ('M1_feat_nograph',{**BASE_NOGRAPH, 'no_graph': True}),
        ('M2_graph',       {**BASE_NOGRAPH}),                       # adds bare graph
        ('M3_cross',       {**M5, 'K':1, 'layers':1, 'tq':TQ_FLAT,  # adds cross-surface
                            'max_edge_age': 0}),
        ('M4_multihop',    {**M5, 'tq':TQ_FLAT, 'max_edge_age': 0}),# adds tuned (K, layers)
        ('M5_bakes',       {**M5}),                                 # adds TQ tiers + age decay
    ]

    # Part B — LOO from M5_full (grouped)
    def _drop(cols): return sorted(set(ZERO_COLS_TIDY) | set(cols))
    STEPS_B = [
        # Node features
        ('loo_node_degree',     'node', {**M5, 'zero_cols': _drop(FEAT_GROUPS['degree'])}),
        # Addition test: static player attributes ON (height/weight/bmi/handedness, cols 0-3).
        # NB this ADDS features rather than removing; +ve "reduction" => static features HURT.
        ('add_node_static',     'node', {**M5, 'zero_cols': sorted(set(ZERO_COLS_TIDY) - {0, 1, 2, 3})}),
        # Edge weights (5)
        ('loo_edge_cross',      'edge', {**M5, 'cross': NO_CROSS}),
        ('loo_edge_cross_sym',  'edge', {**M5, 'cross': SYM_CROSS}),
        ('loo_edge_tq',         'edge', {**M5, 'tq': TQ_FLAT}),
        ('loo_edge_age',        'edge', {**M5, 'max_edge_age': 0}),
        ('loo_edge_uniform_w',  'edge', {**M5, 'uniform_edge_w': True}),
        # Structural (3)
        ('loo_struct_K1',       'struct', {**M5, 'K': 1, 'layers': 1}),
        ('loo_struct_q0',       'struct', {**M5, 'q': 0.0}),
        ('loo_struct_gran_games','struct', {**M5, 'granularity': 'games'}),
    ]

    con.print(f"[bold]Clean ablation (parts={args.part})[/bold]  epochs={EPOCHS}  "
              f"limit={DATA_LIMIT}  json={JSON_PATH}")

    results = []  # (model, gender, brier, n, part, group)

    # ── GNN runs ──────────────────────────────────────────────────────────────
    for gender in ['men', 'women']:
        con.print(f"[bold cyan]{gender.upper()}[/bold cyan]")
        df = pd.read_csv(DATA[gender])
        if DATA_LIMIT: df = df.head(DATA_LIMIT)
        nfm = PlayerInteractionMatrix._create_node_feature_matrix(gender=gender)
        # M0 random-identity vector: seeded N(0,1) per player x len(ID_COLS)
        gen = torch.Generator().manual_seed(SEED)
        id_features = torch.randn((nfm.shape[0], len(ID_COLS)), generator=gen)
        wf = precompute_walk_forward_data(df, gender)

        if args.part in ('A','both'):
            con.print(f"  [dim]Part A — stepwise[/dim]")
            for name, cfg in STEPS_A:
                if args.only and args.only not in name: continue
                t0 = time.time()
                b, n = run_gnn(P, gender, df, wf, nfm, cfg, EPOCHS, id_features=id_features)
                con.print(f"    {name:20s} Brier={b:.5f}  (n={n}, {time.time()-t0:.0f}s)")
                results.append((name, gender, b, n, 'A', '-'))

        if args.part in ('B','both'):
            con.print(f"  [dim]Part B — LOO from M5_full[/dim]")
            for name, group, cfg in STEPS_B:
                if args.only and args.only not in name: continue
                if name == 'add_node_static' and not nfm[:, :4].any():  # attributes not public: skip row
                    results.append((name, gender, np.nan, 0, 'B', group)); continue
                t0 = time.time()
                b, n = run_gnn(P, gender, df, wf, nfm, cfg, EPOCHS, id_features=id_features)
                con.print(f"    {name:22s} [{group:6s}] Brier={b:.5f}  (n={n}, {time.time()-t0:.0f}s)")
                results.append((name, gender, b, n, 'B', group))

    # ── Save ──────────────────────────────────────────────────────────────────
    df_r = pd.DataFrame(results, columns=['model', 'gender', 'brier', 'n', 'part', 'group'])
    if args.append and os.path.exists(OUT_PATH):                       # Splice into existing CSV
        prev = pd.read_csv(OUT_PATH)
        prev = prev[~prev['model'].isin(df_r['model'])]                # drop superseded rows
        df_r = pd.concat([prev, df_r], ignore_index=True)
    skipped = df_r.loc[df_r['brier'].isna(), 'model'].unique()         # rows needing non-public data
    if len(skipped):  # keep the included result rather than overwrite it with NaN
        kept = pd.read_csv(OUT_PATH).query('model in @skipped')
        if len(kept) != 2 * len(skipped): raise ValueError(f'{OUT_PATH} lacks the included results for {list(skipped)}')
        df_r = pd.concat([df_r[~df_r['model'].isin(skipped)], kept], ignore_index=True)
        con.print(f"Kept the included results for {list(skipped)} (player attributes not public)")
    df_r.to_csv(OUT_PATH, index=False)
    con.print(f"Saved -> {OUT_PATH}")
    print_tables(df_r)


def print_tables(df_r):  # Table 3 (pooled across tours by match count) + Part A stepwise diagnostic
    b = df_r.pivot(index='model', columns='gender', values='brier')
    n = df_r.pivot(index='model', columns='gender', values='n')
    b['pooled'] = (b['men'] * n['men'] + b['women'] * n['women']) / (n['men'] + n['women'])
    ref = b.loc['M5_bakes', 'pooled']
    tbl = Table(title=f"Table 3: pooled Brier change vs the full model (reference {ref:.4f}), validation set")
    for c in ['Variant', 'Men', 'Women', 'Pooled', 'dBrier']:
        tbl.add_column(c, justify='right')
    for k, (m, lab) in enumerate(TABLE3):
        r = b.loc[m]
        tbl.add_row(lab + ('*' if m in NOT_PUBLIC else ''), f"{r['men']:.5f}", f"{r['women']:.5f}", f"{r['pooled']:.4f}",
                    f"{abs(r['pooled'] - ref) if round(r['pooled'] - ref, 4) == 0 else r['pooled'] - ref:.4f}",
                    end_section=k == 6)  # 'Remove a component' | 'Simplify or add'
    con.print(tbl)
    con.print("* Needs player attributes from tennisexplorer.com, whose terms do not permit publishing them; "
              "--full keeps the included result.")
    tbl = Table(title="Part A stepwise build-up (diagnostic; only its first row is in Table 3)")
    for c in ['Model', 'Men', 'Women', 'Pooled', 'd vs prev']:
        tbl.add_column(c, justify='right')
    pA = b.reindex(['M0_id_nograph', 'M1_feat_nograph', 'M2_graph', 'M3_cross', 'M4_multihop', 'M5_bakes'])
    for m, r, d in zip(pA.index, pA.itertuples(), pA['pooled'].diff()):
        tbl.add_row(m, f"{r.men:.5f}", f"{r.women:.5f}", f"{r.pooled:.4f}", '-' if pd.isna(d) else f"{d:.4f}")
    con.print(tbl)

if __name__ == '__main__':
    main()
