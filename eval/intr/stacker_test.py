#!/usr/bin/env python3
"""Stacker encompassing test (PER-SLICE variant; not the paper's table generator).

The Global rows fit on the full validation set and match the paper's headline
encompassing result. The IA=0 / IA>0 subset rows below REFIT the stacker on the
matching validation subset (per-slice) -- more permissive than the paper's fixed
scheme, so their subset p-values run smaller. The PUBLISHED subset/threshold tables
(gap_eff, threshold_sens) use the fixed scheme (fit once on full VAL): see
eval/intr/conflict_cycle.py and eval/intr/threshold_sens.py for those numbers.

For each h in {1, 2, 3} (raw-triad H2H min-evidence threshold):
  For each gender in {pooled, men, women}:
    For each slice in {Global, IA=0, IA>0}:
      Report: N, B(Pin), B(WElo_refit), B(W+G stacker), dBrier, DM p,
              beta_GNN (sign), gap-closed %.

Stacker: logit(p) = a0 + a1 logit(WElo) + a2 logit(GNN). Fit on VAL, eval TEST.
Encompassing baseline: WElo-logit refit on VAL (Granger-Ramanathan 1984 frame).
DM one-sided: stacker < refit-WElo (Brier).
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

con = Console(width=150)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
VAL_PREDS  = 'data/predictions/preds_w_ia_val.csv'
RAW_TEST = {1: 'data/predictions/ia/raw_ia.csv',
            2: 'data/predictions/ia/raw_ia_h2.csv',
            3: 'data/predictions/ia/raw_ia_h3.csv'}
RAW_VAL  = {1: 'data/predictions/ia/raw_ia_val.csv',
            2: 'data/predictions/ia/raw_ia_val_h2.csv',
            3: 'data/predictions/ia/raw_ia_val_h3.csv'}
GNN, ELO, PIN = 'model_prob_non_isolated', 'elo_weighted_prob', 'PSA_ip'
EPS = 1e-6
SEED = 8


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(preds_path, raw_path):  # merge on (Date,Aname,Bname,gender_source)
    df = pd.read_csv(preds_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    r = pd.read_csv(raw_path, parse_dates=['Date'])
    df = df.merge(r[['Date','Aname','Bname','gender_source','raw_intransitivity_measure']],
                  on=['Date','Aname','Bname','gender_source'], how='left')
    return df.dropna(subset=[GNN, ELO, PIN, 'Awin', 'raw_intransitivity_measure']).copy()


def fit_stacker(elo_v, gnn_v, y_v, elo_t, gnn_t):
    """Fit logit stacker on VAL; return (p_test, beta_gnn, z_gnn). beta sign is descriptive."""
    X_v = np.column_stack([logit(elo_v), logit(gnn_v)])
    X_t = np.column_stack([logit(elo_t), logit(gnn_t)])
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(X_v, y_v)
    Xd = np.column_stack([np.ones(len(X_v)), X_v])           # design w/ intercept
    pv = m.predict_proba(X_v)[:, 1]
    cov = np.linalg.inv((Xd.T * (pv * (1 - pv))) @ Xd)       # (X'WX)^-1 Fisher info
    z_gnn = m.coef_[0, 1] / np.sqrt(cov[2, 2])               # Wald z, GNN coef = idx 2
    return m.predict_proba(X_t)[:, 1], m.coef_[0, 1], z_gnn


def fit_welo(elo_v, y_v, elo_t):  # encompassing baseline
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs')\
        .fit(logit(elo_v).reshape(-1, 1), y_v)
    return m.predict_proba(logit(elo_t).reshape(-1, 1))[:, 1]


def dm_p(p_new, p_ref, y):  # one-sided: new < ref (Brier)
    d = (p_new - y)**2 - (p_ref - y)**2
    n = len(d); mu, sd = d.mean(), d.std(ddof=1) / np.sqrt(n)
    return np.nan if sd == 0 or n < 10 else 1 - norm.cdf(-mu / sd)


SLICES = [('Global', None),
          ('IA=0',   lambda d: d['raw_intransitivity_measure'] == 0),
          ('IA>0',   lambda d: d['raw_intransitivity_measure'] > 0)]


def run(h):
    test, val = load(TEST_PREDS, RAW_TEST[h]), load(VAL_PREDS, RAW_VAL[h])
    rows = []
    for gender in ['pooled', 'men', 'women']:
        t = test if gender == 'pooled' else test[test['gender_source'] == gender]
        v = val  if gender == 'pooled' else val [val ['gender_source'] == gender]
        for name, fn in SLICES:
            ts = t if fn is None else t[fn(t)]
            vs = v if fn is None else v[fn(v)]
            if len(ts) < 50 or len(vs) < 50: continue
            p_s, beta, z_gnn = fit_stacker(vs[ELO].values, vs[GNN].values, vs['Awin'].values,
                                    ts[ELO].values, ts[GNN].values)
            p_w = fit_welo(vs[ELO].values, vs['Awin'].values, ts[ELO].values)
            y = ts['Awin'].values
            b_pin, b_w, b_s = ((ts[PIN].values - y)**2).mean(), ((p_w-y)**2).mean(), ((p_s-y)**2).mean()
            db = b_s - b_w                                   # negative = stacker better
            gap = b_w - b_pin                                # positive = Pin better
            pct_gap = (-db / gap * 100) if gap > 0 else np.nan
            rows.append({'h': h, 'gender': gender, 'slice': name, 'N': len(ts),
                         'B_pin': b_pin, 'B_welo': b_w, 'B_stack': b_s, 'dB': db,
                         'dm_p': dm_p(p_s, p_w, y), 'beta_gnn': beta, 'z_gnn': z_gnn,
                         'pct_gap': pct_gap})
    return rows


np.random.seed(SEED)
for h in [1, 2, 3]:
    rows = run(h)
    tbl = Table(title=f"WElo-encompassing test, raw-IA h={h}: Global rows fit on full VAL (paper); "
                      f"IA=0 / IA>0 rows refit per slice (diagnostic, not Table 8)")
    for c in ["gender", "slice", "N", "B(Pin)", "B(WElo)", "B(W+G)", "dB", "DM p",
              "b_GNN", "z", "% gap"]:
        tbl.add_column(c, justify="right")
    for i, r in enumerate(rows):
        pg = f"{r['pct_gap']:.1f}" if not np.isnan(r['pct_gap']) else '-'
        tbl.add_row(r['gender'] if r['slice'] == 'Global' else '', r['slice'], f"{r['N']:,}",
                    f"{r['B_pin']:.4f}", f"{r['B_welo']:.4f}", f"{r['B_stack']:.4f}",
                    f"{r['dB']:.4f}", f"{r['dm_p']:.3f}", f"{r['beta_gnn']:.3f}", f"{r['z_gnn']:.2f}", pg,
                    end_section=i + 1 < len(rows) and rows[i + 1]['gender'] != r['gender'])
    con.print(tbl)
