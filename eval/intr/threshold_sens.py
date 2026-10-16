#!/usr/bin/env python3
"""Threshold-sensitivity of the encompassing gain by triad type (Table tab:threshold_sens).

Paper (Giacomini-White fixed) scheme: the WElo-refit baseline and the W+G combination are each
fit ONCE on the full validation set, then evaluated for each raw-triad evidence threshold h in
{1,2,3} on two test cells: intransitive (>=1 cyclic triad) and transitive (triads present, none
cyclic). Same fixed scheme as eval/intr/conflict_cycle.py; contrast eval/intr/stacker_test.py,
whose subset rows REFIT per slice (a separate, more permissive variant).

Per cell: N test matches, dB (Brier reduction vs refit-WElo), eta (share of WElo-to-Pinnacle
headroom closed, Eq. gap_fraction), one-sided Diebold-Mariano p.
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table

con = Console(width=100)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
VAL_PREDS  = 'data/predictions/preds_w_ia_val.csv'
RAW_TEST = {1: 'data/predictions/ia/raw_ia.csv',
            2: 'data/predictions/ia/raw_ia_h2.csv',
            3: 'data/predictions/ia/raw_ia_h3.csv',
            4: 'data/predictions/ia/raw_ia_h4.csv'}
RAW_VAL  = 'data/predictions/ia/raw_ia_val.csv'        # combination fit once on full VAL (h-independent)
GNN, ELO, PIN = 'model_prob_non_isolated', 'elo_weighted_prob', 'PSA_ip'
EPS, SEED = 1e-6, 8


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(preds, raw):  # merge preds with raw-triad counts on (Date,Aname,Bname,gender_source)
    df = pd.read_csv(preds, low_memory=False); df['Date'] = pd.to_datetime(df['Date'])
    r = pd.read_csv(raw, parse_dates=['Date'])
    df = df.merge(r[['Date', 'Aname', 'Bname', 'gender_source', 'n_triads_total', 'n_triads_intransitive']],
                  on=['Date', 'Aname', 'Bname', 'gender_source'], how='left')
    return df.dropna(subset=[GNN, ELO, PIN, 'Awin', 'n_triads_total']).copy()


def fit_lr(Xv, yv, Xt):  # logistic fit on VAL, predict TEST
    return LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(Xv, yv).predict_proba(Xt)[:, 1]


def dm_p(p_new, p_ref, y):  # one-sided Diebold-Mariano on Brier: new < ref
    d = (p_new - y)**2 - (p_ref - y)**2; n = len(d); sd = d.std(ddof=1) / np.sqrt(n)
    return np.nan if sd == 0 or n < 10 else 1 - norm.cdf(-d.mean() / sd)


np.random.seed(SEED)
# Fit W+G combination and WElo-refit ONCE on the full validation set (fixed scheme)
val = load(VAL_PREDS, RAW_VAL)
yv = val['Awin'].values
Xv = np.column_stack([logit(val[ELO]), logit(val[GNN])])

tbl = Table(title="Encompassing gain by triad type (fixed scheme -> Table threshold_sens)")
rows = []
for c in ["h", "Triads", "N", "dB", "eta %", "DM p"]:
    tbl.add_column(c, justify="right")
for h in [1, 2, 3, 4]:
    test = load(TEST_PREDS, RAW_TEST[h])
    # two cells per threshold: intransitive (>=1 cyclic triad) and transitive (triads exist, none cyclic)
    cells = [('Intransitive', test['n_triads_intransitive'] > 0),
             ('Transitive',  (test['n_triads_total'] > 0) & (test['n_triads_intransitive'] == 0))]
    for j, (lab, mask) in enumerate(cells):
        ts = test[mask]
        y = ts['Awin'].values
        Xt = np.column_stack([logit(ts[ELO]), logit(ts[GNN])])
        p_wg = fit_lr(Xv, yv, Xt)                       # W+G combination
        p_w  = fit_lr(Xv[:, :1], yv, Xt[:, :1])         # WElo refit (encompassing baseline)
        b_pin, b_w, b_wg = ((ts[PIN].values - y)**2).mean(), ((p_w - y)**2).mean(), ((p_wg - y)**2).mean()
        dB = b_w - b_wg                                 # positive = combination better
        eta = dB / (b_w - b_pin) * 100 if b_w > b_pin else np.nan
        p = dm_p(p_wg, p_w, y)
        rows.append({'h': h, 'triads': lab.lower(), 'n': len(ts), 'dB': dB, 'eta_pct': eta, 'dm_p': p})
        tbl.add_row(str(h) if j == 0 else "", lab, f"{len(ts):,}", f"{dB:.4f}", f"{eta:.1f}", f"{p:.3f}", end_section=j == 1)
con.print(tbl)
pd.DataFrame(rows).to_csv('outputs/threshold_sens.csv', index=False)
