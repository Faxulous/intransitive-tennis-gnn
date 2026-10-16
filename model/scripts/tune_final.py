#!/usr/bin/env python3
"""Joint tune, cold start (first stage). NSGA-II Pareto over (men_brier, women_brier); 300 trials by default
(the shipped first-stage search, tunes/tune_tqbins_*, holds 500).

Tier-based tournament weighting: GS pinned at 1.0; beta_1000, beta_Finals,
beta_500 each free in [0.0, 1.2]. Granularity in {games, sets, points}.

--variant full          (default): 19-param first-stage grid, all params free.
--variant post-ablation: drops keys that LOO ablation flagged as flat
  (3 cross-surface asymmetry weights CH/GH/GC -> mirror HC/HG/CG; tier
  weights uniform 1.0). Final JSON is filled out so build_preds_w_ia.py
  and GraphEngine consume it unchanged.

Test mode (--test): 2 trials x 5 initial / 2 retrain epochs, 2000-row slice.

Run: python model/scripts/tune_final.py            # full cold tune
     python model/scripts/tune_final.py --test     # quick smoke
"""
import argparse, optuna, sys, os, time, warnings, torch, json
import numpy as np, pandas as pd
from functools import partial
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))   # repo root
from model.graph.player_interaction import PlayerInteractionMatrix
from model.graph.gnn import GraphEngine
from model.walk_forward_validation import walk_forward_validation, precompute_walk_forward_data
from model.seed import set_seed

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
ZERO_COLS = [0, 1, 2, 3, 8]  # static feats (h/w/bmi/righthand) + bp_avg — T203-tidy bake
RETRAIN_EVERY = 38           # retrain every 38 snapshots (fixed, Section 4.3)
LR_MULT = 0.5                # half lr — T203-tidy bake
PRUNED_KEYS = {'CH', 'GH', 'GC',                          # surface asymmetry (post-ablation)
               'beta_1000', 'beta_Finals', 'beta_500'}    # tier weights (post-ablation)

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Quick 2-trial smoke test')
parser.add_argument('--variant', choices=['full', 'post-ablation'], default='full',
                    help='full = 19-param first-stage grid (default); post-ablation = 13-param refined grid')
args = parser.parse_args()

STEM = 'tune_tqbins' if args.variant == 'full' else 'tune_post_ablation'
SUFFIX = '_test' if args.test else ''
if args.test:
    N_TRIALS, INIT_EPOCHS, RETRAIN_EPOCHS, DATA_LIMIT = 2, 5, 2, 2000
else:
    N_TRIALS, INIT_EPOCHS, RETRAIN_EPOCHS, DATA_LIMIT = 300, 100, 30, None
OUT_CSV = f'tunes/{STEM}{SUFFIX}_trials.csv'
OUT_JSON = f'tunes/{STEM}{SUFFIX}_best.json'
# SQLite on local Desktop to avoid Drive-sync state corruption (Optuna asserts on
# orphan RUNNING trials when commits race the sync). Move .db onto Drive at end.
DB_DIR = os.path.expanduser('~/Desktop/optuna_dbs')
os.makedirs(DB_DIR, exist_ok=True)
STORAGE = f'sqlite:///{DB_DIR}/{STEM}{SUFFIX}.db'
STUDY_NAME = f'{STEM}{SUFFIX}'

con = Console()
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)


def apply_tidy_bakes(ge):
    """Zero out feature cols [0,1,2,3,8] each time surface graph rebuilds."""
    orig = ge.pim.create_surface_graph
    def patched(surface, current_time=None, _o=orig, _c=ZERO_COLS):
        _o(surface, current_time)
        for c in _c:
            ge.pim.surface_graphs[surface]['X_real'][:, c] = 0.0
    ge.pim.create_surface_graph = patched


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBJECTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def objective(trial, data_cache=None, nfm_cache=None, wf_cache=None):
    """Minimise (men_brier, women_brier) on VAL-internal test window."""
    t0 = time.time()

    # Architecture
    activation = trial.suggest_categorical('activation', [True, False])
    K = trial.suggest_categorical('K', [1, 2, 3])
    hidden = trial.suggest_categorical('hidden', [16, 32, 64, 128])
    layers = trial.suggest_categorical('layers', [1, 2])

    # Optimisation
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    q = trial.suggest_float('q', 0.1, 0.25)

    # Graph dynamics
    granularity = trial.suggest_categorical('granularity', ['games', 'sets', 'points'])
    lambda_param = trial.suggest_float('lambda_param', 0.0, 2.5)
    max_edge_age = trial.suggest_float('max_edge_age', 0.0, 8.0)
    min_dom_pctile = trial.suggest_float('min_dom_pctile', 0.0, 0.5)

    # Surface cross-weights — full: 6 free (asymmetric); post-ablation: 3 free (symmetric)
    HC = trial.suggest_float('HC', 0.0, 0.5)
    HG = trial.suggest_float('HG', 0.0, 0.5)
    CG = trial.suggest_float('CG', 0.0, 0.5)
    if args.variant == 'full':
        CH = trial.suggest_float('CH', 0.0, 0.5)
        GH = trial.suggest_float('GH', 0.0, 0.5)
        GC = trial.suggest_float('GC', 0.0, 0.5)
    else:
        CH, GH, GC = HC, HG, CG
    surfaces = {
        "Hard":  {"Hard": 1.0, "Clay": HC,  "Grass": HG},
        "Clay":  {"Hard": CH,  "Clay": 1.0, "Grass": CG},
        "Grass": {"Hard": GH,  "Clay": GC,  "Grass": 1.0},
    }

    # Tier weights — GS pinned at 1.0; full: 3 free in [0.0, 1.2]; post-ablation: uniform 1.0.
    if args.variant == 'full':
        tq = {'GS': 1.0,
              '1000':   trial.suggest_float('beta_1000',   0.0, 1.2),
              'Finals': trial.suggest_float('beta_Finals', 0.0, 1.2),
              '500':    trial.suggest_float('beta_500',    0.0, 1.2)}
    else:
        tq = {'GS': 1.0, 'Finals': 1.0, '1000': 1.0, '500': 1.0}

    con.print(f"  T{trial.number}: act={int(activation)} K={K} h={hidden} L={layers} "
              f"lr={lr*LR_MULT:.5f} gran={granularity[:8]}")

    briers = {}
    try:
        for gender in ['men', 'women']:
            t1 = time.time()
            torch.cuda.empty_cache()
            ge = GraphEngine(
                lr=lr*LR_MULT, granularity=granularity, K=K, hidden=hidden, layers=layers,
                activation=activation, gender=gender, surfaces=surfaces,
                lambda_param=lambda_param, tournament_quality=tq,
                max_edge_age=max_edge_age, min_dom_pctile=min_dom_pctile,
                node_feature_matrix=nfm_cache[gender], q=q,
                mlp_head=False, n_edge_features=0,
            )
            apply_tidy_bakes(ge)  # zero_cols bake
            pred_df = walk_forward_validation(
                ge, data_cache[gender], gender, seed=SEED, verbose=False,
                initial_epochs=INIT_EPOCHS, subsequent_epochs=RETRAIN_EPOCHS,
                retrain_every=RETRAIN_EVERY,
                cached_data=wf_cache[gender],
            )
            valid = pred_df['model_prob'].notna() & pred_df['Awin'].notna()
            y = pred_df.loc[valid, 'Awin'].values
            p = pred_df.loc[valid, 'model_prob'].values
            briers[gender] = float(np.mean((p - y) ** 2))
            con.print(f"    [{gender}] {time.time()-t1:.0f}s Brier={briers[gender]:.5f}")
    except Exception as e:
        con.print(f"    T{trial.number} failed: {e}")
        torch.cuda.empty_cache()
        return 0.25, 0.25

    trial.set_user_attr('brier_men', briers['men'])
    trial.set_user_attr('brier_women', briers['women'])
    trial.set_user_attr('activation', activation)
    trial.set_user_attr('granularity', granularity)

    con.print(f"  T{trial.number}: men={briers['men']:.5f} women={briers['women']:.5f} "
              f"avg={(briers['men']+briers['women'])/2:.5f} ({time.time()-t0:.0f}s)")
    return briers['men'], briers['women']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    set_seed(SEED)

    con.print(f"[bold]Joint tune: {N_TRIALS} trials, cold start, tidy baked[/bold]")
    con.print(f"  Baked: zero_cols={ZERO_COLS}, lr_mult={LR_MULT}")
    con.print(f"  Epochs: {INIT_EPOCHS} initial / {RETRAIN_EPOCHS} retrain"
              f"{f' — TEST MODE, data limit {DATA_LIMIT}' if args.test else ''}\n")

    data_cache = {
        'men': pd.read_csv("data/atp_cleaned_VAL.csv"),
        'women': pd.read_csv("data/wta_cleaned_VAL.csv"),
    }
    if DATA_LIMIT:
        data_cache = {g: df.head(DATA_LIMIT) for g, df in data_cache.items()}
    nfm_cache = {
        'men': PlayerInteractionMatrix._create_node_feature_matrix(gender='men'),
        'women': PlayerInteractionMatrix._create_node_feature_matrix(gender='women'),
    }
    wf_cache = {g: precompute_walk_forward_data(data_cache[g], g) for g in ['men', 'women']}
    con.print(f"VAL: men={len(data_cache['men'])}, women={len(data_cache['women'])}\n")

    study = optuna.create_study(
        directions=['minimize', 'minimize'],
        sampler=optuna.samplers.NSGAIISampler(seed=SEED),  # Optuna's multi-objective default, made explicit
        study_name=STUDY_NAME,
        storage=STORAGE, load_if_exists=True,
    )

    remaining = max(0, N_TRIALS - len(study.trials))   # resume-aware: only run trials still needed
    con.print(f"[bold]Study has {len(study.trials)} trials; running {remaining} more to reach {N_TRIALS}[/bold]")
    study.optimize(
        partial(objective, data_cache=data_cache, nfm_cache=nfm_cache, wf_cache=wf_cache),
        n_trials=remaining, n_jobs=1,
    )

    study.trials_dataframe().to_csv(OUT_CSV, index=False)

    # Pareto front
    pareto = study.best_trials
    con.rule(f"[bold]PARETO FRONT ({len(pareto)} non-dominated of {N_TRIALS})[/bold]")
    for t in sorted(pareto, key=lambda x: sum(x.values)):
        con.print(f"  T{t.number}: men={t.values[0]:.5f} women={t.values[1]:.5f} "
                  f"avg={sum(t.values)/2:.5f}")

    # Best min-sum (primary selection criterion)
    best = min(pareto, key=lambda x: sum(x.values))
    con.rule("[bold green]BEST (min sum Brier)[/bold green]")
    con.print(f"  T{best.number}: men={best.values[0]:.5f} women={best.values[1]:.5f} "
              f"avg={sum(best.values)/2:.5f}")
    con.print("\n  Params:")
    for k, v in sorted(best.params.items()):
        con.print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    # Belt + braces: Optuna intermittently drops keys from trial.params (recorded in
    # system_attrs.fixed_params instead).
    base = {'activation','K','hidden','layers','lr','q',
            'granularity','lambda_param','max_edge_age','min_dom_pctile','HC','HG','CG'}
    expected = base | PRUNED_KEYS if args.variant == 'full' else base
    final_params = {**dict(best.params), **best.system_attrs.get('fixed_params', {})}
    missing = expected - set(final_params)
    if missing: raise RuntimeError(f"final_params missing {missing}; check Optuna study state")
    if args.variant == 'post-ablation':                                    # mirror + uniform fill
        final_params['CH'] = final_params['HC']
        final_params['GH'] = final_params['HG']
        final_params['GC'] = final_params['CG']
        for k in ['beta_1000', 'beta_Finals', 'beta_500']:
            final_params[k] = 1.0

    with open(OUT_JSON, 'w') as f:
        json.dump({
            'trial_number': best.number,
            'brier_men_val': best.values[0],
            'brier_women_val': best.values[1],
            'params': final_params,
        }, f, indent=2)
    con.print(f"\n[bold]Best params -> {OUT_JSON}[/bold]")
    con.print(f"All trials -> {OUT_CSV}")

    # Manifest: one row per tune run for cross-run audit / Pareto comparison
    manifest_path = 'tunes/runs.csv'
    row = pd.DataFrame([{
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'variant': args.variant, 'study_name': STUDY_NAME,
        'n_trials': N_TRIALS, 'init_epochs': INIT_EPOCHS, 'retrain_epochs': RETRAIN_EPOCHS,
        'best_trial': best.number,
        'brier_men_val': best.values[0], 'brier_women_val': best.values[1],
        'brier_avg_val': sum(best.values) / 2,
        'pareto_size': len(pareto),
        'trials_csv': OUT_CSV, 'best_json': OUT_JSON,
    }])
    row.to_csv(manifest_path, mode='a', header=not os.path.exists(manifest_path), index=False)
    con.print(f"Manifest -> {manifest_path}")
