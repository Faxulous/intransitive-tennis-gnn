#!/usr/bin/env python3
"""Scalar-family encompassing: the GNN's orthogonal signal is specifically non-scalar.

Paper-facing numbers (Section 6.2, Partner-uniqueness table): no scalar partner adds
information to refit-WElo on TEST (surface Elo p = 0.28, Bradley-Terry p = 0.78; all-surface
Elo is also insignificant), while the GNN does. The family stacks (S4 = all four scalars,
S4+G = S4 plus the GNN) and their robustness checks are diagnostics, not in the paper.
All weights fit once on VAL (paper scheme), evaluated on TEST.

Run: python eval/intr/scalar_family.py
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
SEED, EPS = 8, 1e-6
TEST_PREDS, VAL_PREDS = 'data/predictions/preds_w_ia.csv', 'data/predictions/preds_w_ia_val.csv'
W, ES, EA, BT, G = 'elo_weighted_prob', 'elo_surf_prob', 'elo_all_prob', 'bt_prob', 'model_prob_non_isolated'
PIN = 'PSA_ip'
NM = {W: 'W', ES: 'Es', EA: 'Ea', BT: 'BT', G: 'G'}


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(path):
    df = pd.read_csv(path, low_memory=False)
    return df.dropna(subset=[W, ES, EA, BT, G, PIN, 'Awin']).reset_index(drop=True)


def stack(cols, v, t):  # fit on full VAL, predict TEST; returns (p_test, coefs, Wald z)
    Xv = np.column_stack([logit(v[c].values) for c in cols])
    Xt = np.column_stack([logit(t[c].values) for c in cols])
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs', max_iter=1000).fit(Xv, v['Awin'].values)
    Xd = np.column_stack([np.ones(len(Xv)), Xv]); pv = m.predict_proba(Xv)[:, 1]
    se = np.sqrt(np.diag(np.linalg.inv((Xd.T * (pv * (1-pv))) @ Xd)))[1:]
    return m.predict_proba(Xt)[:, 1], m.coef_[0], m.coef_[0] / se


def dm_p(p_new, p_ref, y):  # one-sided DM on Brier: new < ref
    d = (p_new - y)**2 - (p_ref - y)**2
    mu, sd = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return np.nan if sd == 0 or len(d) < 10 else 1 - norm.cdf(-mu / sd)


def table(title, cols, rows):  # rich table, all columns right-aligned
    t = Table(title=title)
    for c in cols: t.add_column(c, justify='right')
    for r in rows: t.add_row(*r)
    con.print(t)


def main():
    np.random.seed(SEED)
    t, v = load(TEST_PREDS), load(VAL_PREDS)
    y = t['Awin'].values
    t['Date'] = pd.to_datetime(t['Date'])

    P, fam = {}, []
    for tag, cols in [('refitW', [W]), ('W+G', [W, G]), ('S4', [W, ES, EA, BT]), ('S4+G', [W, ES, EA, BT, G])]:
        p, coef, z = stack(cols, v, t); P[tag] = p
        cz = {c: f'{b:.3f} ({zz:.2f})' for c, b, zz in zip(cols, coef, z)}
        fam.append((tag, f'{((p-y)**2).mean():.5f}', *(cz[c] if c in cz else '-' for c in [W, ES, EA, BT, G])))
    table('Family stacks on TEST (weights fit once on full VAL); coefficient (z)',
          ['Stack', 'Brier', *(NM[c] for c in [W, ES, EA, BT, G])], fam)
    rows = []
    for new, ref in [('S4', 'refitW'), ('W+G', 'refitW'), ('S4+G', 'S4'), ('S4+G', 'W+G'), ('W+G', 'S4')]:
        dB = ((P[new]-y)**2).mean() - ((P[ref]-y)**2).mean()
        rows.append((new, ref, f'{dB:.5f}', f'{dm_p(P[new], P[ref], y):.2e}'))
    table('One-sided DM tests between stacks (new < reference)', ['New', 'Reference', 'dB', 'DM p'], rows)

    rows = []
    for X in [ES, EA, BT, G]:
        p, coef, z = stack([W, X], v, t)
        dB = ((p-y)**2).mean() - ((P['refitW']-y)**2).mean()
        rows.append((f'W+{NM[X]}', f'{dB:.5f}', f'{dm_p(p, P["refitW"], y):.2e}', f'{coef[1]:.3f}', f'{z[1]:.2f}'))
    table('Partner-uniqueness for WElo: B(W+X) vs B(refitW)', ['Stack', 'dB', 'DM p', 'b(X)', 'z'], rows)

    d = (P['S4+G']-y)**2 - (P['S4']-y)**2
    cuts = [('men', (t['gender_source'] == 'men').values), ('women', (t['gender_source'] == 'women').values),
            ('era1', (t['Date'] <= t['Date'].median()).values), ('era2', (t['Date'] > t['Date'].median()).values)]
    rows = [(tag, f'{m.sum():,}', f'{d[m].mean():.5f}', f'{dm_p(P["S4+G"][m], P["S4"][m], y[m]):.2e}') for tag, m in cuts]
    rng = np.random.default_rng(SEED); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    table(f'Robustness of S4 -> S4+G gain (bootstrap 95% CI for dB: [{np.percentile(bs, 2.5):.5f}, '
          f'{np.percentile(bs, 97.5):.5f}])', ['Subset', 'N', 'dB', 'DM p'], rows)


if __name__ == '__main__':
    main()
