#!/usr/bin/env python3
"""Compare raw-triad IA direction basis: matches vs sets vs games.

For each basis, bin matches by T_h (>0 vs =0) and run the WElo-encompassing test
(refit-WElo baseline vs WElo+GNN stacker; DM one-sided). The "best" basis is the
one that most cleanly localises the GNN's complementary signal on intransitive
matches: lower DM p / higher gap-efficiency eta on T>0 relative to T=0.
The stacker is REFIT on each validation slice (as in stacker_test.py), so subset
p-values differ from the paper's fixed scheme (Table 8, threshold_sens.py); this
table is a diagnostic. The cross-basis agreement on the T>0 label is in Section 6.3.

Run: python eval/intr/triad_granularity_compare.py
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table

con = Console(width=160)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

TEST_PREDS, VAL_PREDS = 'data/predictions/preds_w_ia.csv', 'data/predictions/preds_w_ia_val.csv'
BASES = {'matches': ('raw_ia.csv', 'raw_ia_val.csv'),
         'sets':    ('raw_ia_sets.csv', 'raw_ia_val_sets.csv'),
         'games':   ('raw_ia_games.csv', 'raw_ia_val_games.csv')}
GNN, ELO, PIN = 'model_prob_non_isolated', 'elo_weighted_prob', 'PSA_ip'
EPS = 1e-6


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(preds, rawfile):
    df = pd.read_csv(preds, low_memory=False); df['Date'] = pd.to_datetime(df['Date'])
    r = pd.read_csv(f'data/predictions/ia/{rawfile}', parse_dates=['Date'])
    df = df.merge(r[['Date','Aname','Bname','gender_source','raw_intransitivity_measure']],
                  on=['Date','Aname','Bname','gender_source'], how='left')
    return df.dropna(subset=[GNN, ELO, PIN, 'Awin', 'raw_intransitivity_measure']).copy()


def fit_stacker(ev, gv, yv, et, gt):
    Xv = np.column_stack([logit(ev), logit(gv)]); Xt = np.column_stack([logit(et), logit(gt)])
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(Xv, yv)
    Xd = np.column_stack([np.ones(len(Xv)), Xv]); pv = m.predict_proba(Xv)[:, 1]
    cov = np.linalg.inv((Xd.T * (pv*(1-pv))) @ Xd)
    return m.predict_proba(Xt)[:, 1], m.coef_[0,1], m.coef_[0,1]/np.sqrt(cov[2,2])


def fit_welo(ev, yv, et):
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs').fit(logit(ev).reshape(-1,1), yv)
    return m.predict_proba(logit(et).reshape(-1,1))[:, 1]


def dm_p(pn, pr, y):
    d = (pn-y)**2 - (pr-y)**2; n = len(d); mu, sd = d.mean(), d.std(ddof=1)/np.sqrt(n)
    return np.nan if sd == 0 or n < 10 else 1 - norm.cdf(-mu/sd)


def slice_row(ts, vs):
    p_s, b2, z = fit_stacker(vs[ELO].values, vs[GNN].values, vs['Awin'].values, ts[ELO].values, ts[GNN].values)
    p_w = fit_welo(vs[ELO].values, vs['Awin'].values, ts[ELO].values)
    y = ts['Awin'].values
    b_pin, b_w, b_s = ((ts[PIN].values-y)**2).mean(), ((p_w-y)**2).mean(), ((p_s-y)**2).mean()
    db, gap = b_s-b_w, b_w-((ts[PIN].values-y)**2).mean()
    eta = (-db/gap*100) if gap > 0 else np.nan
    return len(ts), db, dm_p(p_s, p_w, y), b2, z, eta


tbl = Table(title="Encompassing by triad-direction basis (pooled, h=1; refit per slice, diagnostic)")
for c in ["basis", "slice", "N", "dB", "DM p", "b_GNN", "z", "eta % (gap closed)"]:
    tbl.add_column(c, justify="right")
labels = {}
for basis, (rt, rv) in BASES.items():
    test, val = load(TEST_PREDS, rt), load(VAL_PREDS, rv)
    labels[basis] = (test.set_index(['Date','Aname','Bname','gender_source'])['raw_intransitivity_measure'] > 0)
    for name, mask in [('Global', None),
                       ('T=0', lambda d: d['raw_intransitivity_measure'] == 0),
                       ('T>0', lambda d: d['raw_intransitivity_measure'] > 0)]:
        ts = test if mask is None else test[mask(test)]
        vs = val  if mask is None else val[mask(val)]
        N, db, p, b2, z, eta = slice_row(ts, vs)
        tbl.add_row(basis if name == 'Global' else '', name, f"{N:,}", f"{db:.4f}", f"{p:.3f}",
                    f"{b2:.3f}", f"{z:.2f}", f"{eta:.1f}" if not np.isnan(eta) else '-', end_section=name == 'T>0')
con.print(tbl)

# Cross-basis agreement on T>0 label (aligned on common index)
idx = labels['matches'].index
agree = Table(title="Cross-basis agreement on the T>0 label (test set)")
for c in ["pair", "agree %", "both T>0", "neither", "disagree"]:
    agree.add_column(c, justify="right")
for a, b in [('matches','sets'), ('matches','games'), ('sets','games')]:
    la, lb = labels[a].reindex(idx), labels[b].reindex(idx)
    m = la.notna() & lb.notna(); la, lb = la[m].astype(bool), lb[m].astype(bool)
    agree.add_row(f"{a} vs {b}", f"{(la==lb).mean()*100:.1f}",
                  f"{(la&lb).sum():,}", f"{(~la&~lb).sum():,}", f"{(la!=lb).sum():,}")
con.print(agree)
