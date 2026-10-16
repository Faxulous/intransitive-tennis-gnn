#!/usr/bin/env python3
"""Pinnacle-encompassing test: does the GNN carry information beyond the bookmaker?

Mirror of stacker_test.py but with PINNACLE as the encompassing baseline instead of WElo.
  Baseline  : refit Pin-logit on VAL  (Platt recalibration of the market)
  Alternative: Pin-logit + GNN-logit stacker on VAL
  DM one-sided: stacker < refit-Pin (Brier), HLN not applied (large N, iid-ish).
If b_GNN is significant the market does NOT fully subsume the GNN; if null, Pinnacle
encompasses the GNN and no market claim is supportable. Reported per h, gender, IA slice.

Run: python eval/intr/pin_encompassing.py
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

TEST_PREDS = 'data/predictions/preds_w_ia.csv'
VAL_PREDS  = 'data/predictions/preds_w_ia_val.csv'
RAW_TEST = {1: 'data/predictions/ia/raw_ia.csv', 2: 'data/predictions/ia/raw_ia_h2.csv', 3: 'data/predictions/ia/raw_ia_h3.csv'}
RAW_VAL  = {1: 'data/predictions/ia/raw_ia_val.csv', 2: 'data/predictions/ia/raw_ia_val_h2.csv', 3: 'data/predictions/ia/raw_ia_val_h3.csv'}
GNN, PIN = 'model_prob_non_isolated', 'PSA_ip'
EPS, SEED = 1e-6, 8


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(preds_path, raw_path):
    df = pd.read_csv(preds_path, low_memory=False); df['Date'] = pd.to_datetime(df['Date'])
    r = pd.read_csv(raw_path, parse_dates=['Date'])
    df = df.merge(r[['Date','Aname','Bname','gender_source','raw_intransitivity_measure']],
                  on=['Date','Aname','Bname','gender_source'], how='left')
    return df.dropna(subset=[GNN, PIN, 'Awin', 'raw_intransitivity_measure']).copy()


def fit_stacker(base_v, gnn_v, y_v, base_t, gnn_t):  # Pin + GNN; baseline = market
    X_v = np.column_stack([logit(base_v), logit(gnn_v)])
    X_t = np.column_stack([logit(base_t), logit(gnn_t)])
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(X_v, y_v)
    Xd = np.column_stack([np.ones(len(X_v)), X_v]); pv = m.predict_proba(X_v)[:, 1]
    cov = np.linalg.inv((Xd.T * (pv * (1 - pv))) @ Xd)
    return m.predict_proba(X_t)[:, 1], m.coef_[0, 1], m.coef_[0, 1] / np.sqrt(cov[2, 2])


def fit_base(base_v, y_v, base_t):  # refit-Pinnacle baseline
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(logit(base_v).reshape(-1, 1), y_v)
    return m.predict_proba(logit(base_t).reshape(-1, 1))[:, 1]


def dm_p(p_new, p_ref, y):  # one-sided: new < ref
    d = (p_new - y)**2 - (p_ref - y)**2; n = len(d); mu, sd = d.mean(), d.std(ddof=1) / np.sqrt(n)
    return np.nan if sd == 0 or n < 10 else 1 - norm.cdf(-mu / sd)


SLICES = [('Global', None), ('IA=0', lambda d: d['raw_intransitivity_measure'] == 0),
          ('IA>0', lambda d: d['raw_intransitivity_measure'] > 0)]


def run(h):
    test, val = load(TEST_PREDS, RAW_TEST[h]), load(VAL_PREDS, RAW_VAL[h])
    rows = []
    for gender in ['pooled', 'men', 'women']:
        t = test if gender == 'pooled' else test[test['gender_source'] == gender]
        v = val  if gender == 'pooled' else val [val ['gender_source'] == gender]
        for name, fn in SLICES:
            ts, vs = (t if fn is None else t[fn(t)]), (v if fn is None else v[fn(v)])
            if len(ts) < 50 or len(vs) < 50: continue
            p_s, beta, z = fit_stacker(vs[PIN].values, vs[GNN].values, vs['Awin'].values, ts[PIN].values, ts[GNN].values)
            p_b = fit_base(vs[PIN].values, vs['Awin'].values, ts[PIN].values)
            y = ts['Awin'].values
            b_b, b_s = ((p_b-y)**2).mean(), ((p_s-y)**2).mean()
            rows.append({'gender': gender, 'slice': name, 'N': len(ts), 'B_pin': b_b,
                         'B_pin_gnn': b_s, 'dB': b_s-b_b, 'dm_p': dm_p(p_s, p_b, y),
                         'beta_gnn': beta, 'z_gnn': z})
    return rows


np.random.seed(SEED)
for h in [1, 2, 3]:
    tbl = Table(title=f"Pinnacle-encompassing test, raw-IA h={h}")
    for c in ["gender", "slice", "N", "B(Pin)", "B(Pin+G)", "dB", "DM p", "b_GNN", "z"]:
        tbl.add_column(c, justify="right")
    rows = run(h)
    for i, r in enumerate(rows):
        tbl.add_row(r['gender'] if r['slice'] == 'Global' else '', r['slice'], f"{r['N']:,}", f"{r['B_pin']:.4f}",
                    f"{r['B_pin_gnn']:.4f}", f"{r['dB']:.4f}", f"{r['dm_p']:.3f}", f"{r['beta_gnn']:.3f}", f"{r['z_gnn']:.2f}",
                    end_section=i + 1 < len(rows) and rows[i + 1]['gender'] != r['gender'])
    con.print(tbl)
