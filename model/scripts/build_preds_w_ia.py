#!/usr/bin/env python3
"""Rebuild preds_w_ia.csv + preds_w_ia_val.csv under the canonical config.

Loads tunes/tune_degree_postab_best.json (T170: symmetric surfaces, uniform tournament
tiers — the configuration reported in method.tex Table 2). Node features are
DEGREE-ONLY: only the graph-derived in/out degree (cols 4,5) are kept; the static
player attributes (0-3) and all on-court serve/return statistics (6-13) are zeroed.
The ablation (Section 4.5) shows degree-only matches the full feature set, so the model
uses only data available from tennis-data.co.uk (match results -> dominance graph).
ZERO_COLS therefore keeps {4,5}; LR_MULT=0.5; 100 initial / 30 retrain epochs.

Same params for both genders (joint tune); zeroing is applied at the CLASS level so
training and prediction see the same ablated graph (see _ORIG_CSG patch below).

Outputs:
  data/predictions/preds_w_ia.csv      (held-out TEST, 2022-02 -> 2025-10)
  data/predictions/preds_w_ia_val.csv  (VAL, used to fit the encompassing combinations)

Run: python model/scripts/build_preds_w_ia.py           # both splits, both genders
     python model/scripts/build_preds_w_ia.py --val     # VAL only
     python model/scripts/build_preds_w_ia.py --test    # TEST only
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model.graph.gnn import GraphEngine
from model.graph.player_interaction import PlayerInteractionMatrix
from model.walk_forward_validation import walk_forward_validation
from model.seed import set_seed

_ORIG_CSG = PlayerInteractionMatrix.create_surface_graph   # original class method, restored after each run

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
# Degree-only node features: keep cols 4,5 (in/out degree); zero static 0-3 + all
# 8 on-court stats 6-13. Model relies on graph structure + degree alone (tennis-data only).
ZERO_COLS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
LR_MULT = 0.5                          # lr_half bake
INIT_EPOCHS, RETRAIN_EPOCHS, RETRAIN_EVERY = 100, 30, 38  # epochs (initial, retrain); retrain every 38 snapshots
DEFAULT_JSON = 'tunes/tune_degree_postab_best.json'   # final degree-only tune (T170, 250-trial)
OUT_TEST = 'data/predictions/preds_w_ia.csv'
OUT_VAL  = 'data/predictions/preds_w_ia_val.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--val',  action='store_true', help='Build VAL only')
parser.add_argument('--test', action='store_true', help='Build TEST only')
parser.add_argument('--json', default=DEFAULT_JSON, help=f'Joint config JSON (default: {DEFAULT_JSON})')
parser.add_argument('--out-test', default=OUT_TEST, help='Custom TEST output path')
parser.add_argument('--out-val',  default=OUT_VAL,  help='Custom VAL output path')
args = parser.parse_args()
BUILD_VAL  = args.val  or not args.test
BUILD_TEST = args.test or not args.val
JSON_PATHS = {'men': args.json, 'women': args.json}   # joint config: same params both genders

con = Console()


def build_kw(p):  # Per-gender params → GraphEngine kwargs (LR_MULT baked)
    surfaces = {
        "Hard":  {"Hard": 1.0,    "Clay": p['HC'], "Grass": p['HG']},
        "Clay":  {"Hard": p['CH'], "Clay": 1.0,    "Grass": p['CG']},
        "Grass": {"Hard": p['GH'], "Clay": p['GC'], "Grass": 1.0},
    }
    tq = {'GS': 1.0, 'Finals': p['beta_Finals'],
          '1000': p['beta_1000'], '500': p['beta_500']}
    kw = {
        'lr': p['lr'] * LR_MULT, 'K': p['K'], 'hidden': p['hidden'], 'layers': p['layers'],
        'activation': p['activation'], 'q': p['q'],
        'granularity': p['granularity'], 'lambda_param': p['lambda_param'],
        'max_edge_age': p['max_edge_age'], 'min_dom_pctile': p['min_dom_pctile'],
        'surfaces': surfaces, 'tournament_quality': tq,
        'mlp_head': False, 'n_edge_features': 0,
    }
    return kw


def make_zero_cols_patch(cols=ZERO_COLS):  # CLASS-level patch: zero feat cols after each rebuild
    # Patch the class method (takes self) so the val_pim copy used for TRAINING is ablated too —
    # not just pred_pim. Instance-patching trained on un-ablated features (a train/serve mismatch).
    def csg(self, surface, current_time=None, _o=_ORIG_CSG, _c=cols):
        _o(self, surface, current_time)
        for c in _c:
            self.surface_graphs[surface]['X_real'][:, c] = 0.0
    return csg


def run(gender, kw, use_val):  # Walk-forward + IA for one gender × split
    set_seed(SEED)
    torch.cuda.empty_cache()
    prefix, suffix = ("atp" if gender == "men" else "wta"), ("_VAL" if use_val else "")
    df = pd.read_csv(f"data/{prefix}_cleaned{suffix}.csv")
    ge = GraphEngine(**kw, gender=gender)
    try:
        PlayerInteractionMatrix.create_surface_graph = make_zero_cols_patch()  # train + predict both zeroed
        pred = walk_forward_validation(
            ge, df, gender, seed=SEED, verbose=False,
            initial_epochs=INIT_EPOCHS, subsequent_epochs=RETRAIN_EPOCHS,
            retrain_every=RETRAIN_EVERY, compute_intransitivity=True,
        )
    finally:
        PlayerInteractionMatrix.create_surface_graph = _ORIG_CSG
    pred = pred.rename(columns={'model_prob': 'model_prob_non_isolated'})
    pred['gender_source'] = gender
    if 'Date' in pred.columns:
        pred['Date']  = pd.to_datetime(pred['Date'])
        pred['year']  = pred['Date'].dt.year
        pred['month'] = pred['Date'].dt.month
    return pred


def add_scaled_ia(df):  # Three confidence-scaled IA variants (parity with legacy run.py output)
    im, c = df['intransitivity_measure'].values, df['matchup_confidence'].values
    mask = c > 0
    for col, fn in [('int_attempt1_linear', lambda m, c_: m * c_),
                    ('int_attempt2_sqrt',   lambda m, c_: m * np.sqrt(c_)),
                    ('int_attempt3_log',    lambda m, c_: m * np.log1p(c_))]:
        df[col] = 0.0
        df.loc[mask, col] = fn(im[mask], c[mask])
    return df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    cfgs = {}
    for g in ['men', 'women']:
        with open(JSON_PATHS[g]) as f:
            best = json.load(f)
        kw = build_kw(best['params'])
        cfgs[g] = kw
        con.print(f"[bold]{g.upper()}[/bold] T{best['trial_number']}: Brier_val={best['brier_'+g+'_val']:.5f}"
                  f"  K={kw['K']}  h={kw['hidden']}  L={kw['layers']}"
                  f"  gran={kw['granularity']}  (cfg: {JSON_PATHS[g]})")

    for label, use_val, out_path, enabled in [
        ('VAL',  True,  args.out_val,  BUILD_VAL),
        ('TEST', False, args.out_test, BUILD_TEST),
    ]:
        if not enabled:
            continue
        con.print(f"[bold cyan]Building {label}[/bold cyan]")
        parts = []
        for g in ['men', 'women']:
            t0 = time.time()
            p = run(g, cfgs[g], use_val=use_val)
            con.print(f"  {g}: n={len(p)}  ({time.time()-t0:.0f}s)")
            parts.append(p)
        combined = pd.concat(parts, ignore_index=True)
        if 'Date' in combined.columns:
            combined = combined.sort_values('Date').reset_index(drop=True)
        combined = add_scaled_ia(combined)
        combined.to_csv(out_path, index=False)
        con.print(f"[bold green]-> {out_path}[/bold green]  ({len(combined)} matches)\n")
