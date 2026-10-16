#!/usr/bin/env python3
"""Rating-evidence conflicts and intransitivity: paper-facing numbers (Section 6).

Three results, all model-free conditioning, verified on VAL and TEST:
  S2  Conflicts (WElo favourite contradicts the prior H2H match-majority leader)
      concentrate almost exclusively on cyclic matchups: P(conflict|cyclic)~0.30
      vs ~0.07 acyclic; ~98% of conflicts sit on a T>0 matchup (h=1).
  S3  No recoverable focal-edge signal: the H2H leader wins conflicts at the
      rating-implied rate; adding a lead indicator to logit(WElo) is n.s.;
      the GNN sides with the leader on conflicts as rarely as Pinnacle.
  S4  Encompassing gain is broad-based: significant on the no-common-opponent
      cell (largest dB), the triad-unmeasurable bulk, and T>0; not gated by
      local cycle structure. Paper scheme throughout (weights fit on full VAL).

Run: python eval/intr/conflict_cycle.py
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
import numpy as np, pandas as pd
import statsmodels.api as sm
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
PREDS = {'test': 'data/predictions/preds_w_ia.csv', 'val': 'data/predictions/preds_w_ia_val.csv'}
RAW = {('test', 'h1'): 'data/predictions/ia/raw_ia.csv',       ('val', 'h1'): 'data/predictions/ia/raw_ia_val.csv',
       ('test', 'h2'): 'data/predictions/ia/raw_ia_h2.csv',    ('val', 'h2'): 'data/predictions/ia/raw_ia_val_h2.csv',
       ('test', 'h3'): 'data/predictions/ia/raw_ia_h3.csv',    ('val', 'h3'): 'data/predictions/ia/raw_ia_val_h3.csv',
       ('test', 'games'): 'data/predictions/ia/raw_ia_games.csv', ('val', 'games'): 'data/predictions/ia/raw_ia_val_games.csv',
       ('test', 'sets'): 'data/predictions/ia/raw_ia_sets.csv',   ('val', 'sets'): 'data/predictions/ia/raw_ia_val_sets.csv'}
HIST = {('men', 'test'): 'data/atp_cleaned.csv', ('women', 'test'): 'data/wta_cleaned.csv',
        ('men', 'val'): 'data/atp_cleaned_VAL.csv', ('women', 'val'): 'data/wta_cleaned_VAL.csv'}
GNN, ELO, PIN = 'model_prob_non_isolated', 'elo_weighted_prob', 'PSA_ip'
KEY = ['Date', 'Aname', 'Bname', 'gender_source']
IA_COLS = ['raw_intransitivity_measure', 'n_common_opponents', 'n_triads_total', 'n_triads_intransitive']


def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))


def load(split):
    """Preds merged with all five raw-IA variants (columns suffixed __h1.., __sets)."""
    df = pd.read_csv(PREDS[split], low_memory=False); df['Date'] = pd.to_datetime(df['Date'])
    for tag in ['h1', 'h2', 'h3', 'games', 'sets']:
        r = pd.read_csv(RAW[(split, tag)], parse_dates=['Date'])
        df = df.merge(r[KEY + IA_COLS].rename(columns={c: f'{c}__{tag}' for c in IA_COLS}), on=KEY, how='left')
    df = df.dropna(subset=[GNN, ELO, PIN, 'Awin', 'raw_intransitivity_measure__h1'])
    return df.reset_index(drop=True)


def h2h_pass(df, split):
    """Causal streaming pass: prior A-B H2H match wins for each query (Date < t)."""
    out = np.zeros((len(df), 2), dtype=np.int32)
    for gender in ['men', 'women']:
        hist = pd.read_csv(HIST[(gender, split)], usecols=['Date', 'Aname', 'Bname', 'Awin'],
                           parse_dates=['Date'], low_memory=False).sort_values('Date', kind='mergesort')
        hd, hA, hB, hw = hist['Date'].values, hist['Aname'].values, hist['Bname'].values, hist['Awin'].values
        q = df[df['gender_source'] == gender].sort_values('Date', kind='mergesort')
        w, hi = defaultdict(int), 0
        for qi, (t, A, B) in zip(q.index, q[['Date', 'Aname', 'Bname']].itertuples(index=False)):
            while hi < len(hd) and hd[hi] < t:
                w[(hA[hi], hB[hi]) if hw[hi] else (hB[hi], hA[hi])] += 1; hi += 1
            out[qi] = [w[(A, B)], w[(B, A)]]
    df['h2h_w_A'], df['h2h_w_B'] = out[:, 0], out[:, 1]
    df['h2h_n'] = df['h2h_w_A'] + df['h2h_w_B']
    return df


def masks(df):
    """(conflict, directed): WElo favourite vs prior H2H match-majority leader."""
    lead = np.sign(df['h2h_w_A'].values - df['h2h_w_B'].values)
    pick = np.sign(df[ELO].values - 0.5)
    directed = (df['h2h_n'].values >= 1) & (lead != 0) & (pick != 0)
    return directed & (pick == -lead), directed


def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    z = (k1/n1 - k2/n2) / np.sqrt(p * (1-p) * (1/n1 + 1/n2))
    return z, 2 * (1 - norm.cdf(abs(z)))


def fit_lr(X_v, y_v, X_t):  # unregularised logistic, fit VAL predict TEST
    m = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs', max_iter=1000).fit(X_v, y_v)
    return m.predict_proba(X_t)[:, 1]


def dm_p(p_new, p_ref, y):  # one-sided DM on Brier: new < ref
    d = (p_new - y)**2 - (p_ref - y)**2
    mu, sd = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return np.nan if sd == 0 or len(d) < 10 else 1 - norm.cdf(-mu / sd)


def table(title, cols, rows):  # rich table, all columns right-aligned
    t = Table(title=title)
    for c in cols: t.add_column(c, justify='right')
    for r in rows: t.add_row(*r[:-1] if isinstance(r[-1], bool) else r, end_section=r[-1] is True)
    con.print(t)


def slice_rows(title, test, val, cells):
    """Paper-scheme encompassing per cell: stacker & refit-WElo fit once on full VAL."""
    Xv = np.column_stack([logit(val[ELO].values), logit(val[GNN].values)])
    Xt = np.column_stack([logit(test[ELO].values), logit(test[GNN].values)])
    p_s = fit_lr(Xv, val['Awin'].values, Xt)
    p_w = fit_lr(Xv[:, :1], val['Awin'].values, Xt[:, :1])
    rows = []
    for tag, m in cells:
        y = test['Awin'].values[m]
        bw, bs, bp = ((p_w[m]-y)**2).mean(), ((p_s[m]-y)**2).mean(), ((test[PIN].values[m]-y)**2).mean()
        eta = (bw - bs) / (bw - bp) if bw > bp else np.nan
        rows.append((tag, f'{m.sum():,}', f'{bw:.4f}', f'{bs:.4f}', f'{bp:.4f}', f'{(bs-bw)*1e3:.2f}',
                     f'{dm_p(p_s[m], p_w[m], y):.4f}', f'{eta*100:.1f}'))
    table(title, ['Cell', 'N', 'B(W)', 'B(W+G)', 'B(Pin)', 'dB x1e3', 'DM p', 'eta %'], rows)


def main():
    np.random.seed(SEED)
    data = {s: h2h_pass(load(s), s) for s in ['val', 'test']}

    t, v = data['test'], data['val']
    slice_rows('S1  Sanity: full-test paper headline', t, v, [('full TEST', np.ones(len(t), bool))])

    link, share, lgt = [], [], []  # S2 rows
    for split in ['val', 'test']:
        df = data[split]
        cf, directed = masks(df)
        variants = ['h1'] if split == 'val' else ['h1', 'h2', 'h3', 'games', 'sets']
        for hv in variants:
            Tpos, meas = df[f'n_triads_intransitive__{hv}'].values > 0, df[f'n_triads_total__{hv}'].values > 0
            for lo, hi, tag in [(1, 999, 'all depths'), (1, 1, 'h2h_n=1'), (2, 2, 'h2h_n=2'), (3, 999, 'h2h_n>=3')]:
                d = directed & (df['h2h_n'].values >= lo) & (df['h2h_n'].values <= hi)
                m1, m0 = d & meas & Tpos, d & meas & ~Tpos
                k1, n1, k0, n0 = (cf & m1).sum(), m1.sum(), (cf & m0).sum(), m0.sum()
                lab = (split.upper() if tag == "all depths" else "",
                       hv if tag == 'all depths' else '', tag)
                if min(n1, n0) < 20:  # too few to test: counts only
                    link.append((*lab, f'{k1:,}/{n1:,}', '-', f'{k0:,}/{n0:,}', '-', '-', '-', tag == 'h2h_n>=3'))
                    continue
                z, p = two_prop_z(k1, n1, k0, n0)
                link.append((*lab, f'{n1:,}', f'{k1/n1:.3f}', f'{n0:,}', f'{k0/n0:.3f}', f'{z:.2f}', f'{p:.1e}', tag == 'h2h_n>=3'))
            share.append((split.upper(), hv, f'{(cf & Tpos).sum():,}', f'{cf.sum():,}', f'{(cf & Tpos).sum()/cf.sum():.3f}'))
        # cyclic-rate logistic with density/depth/gender controls (measurable directed)
        m = directed & (df['n_triads_total__h1'].values > 0)
        X = sm.add_constant(pd.DataFrame({
            'cyc_rate': df['raw_intransitivity_measure__h1'].values[m],
            'log1p_ntot': np.log1p(df['n_triads_total__h1'].values[m]),
            'log1p_h2h': np.log1p(df['h2h_n'].values[m]),
            'women': (df['gender_source'].values[m] == 'women').astype(float)}))
        r = sm.Logit(cf[m].astype(float), X).fit(disp=0)
        lgt.append((split.upper(), f'{m.sum():,}', f'{r.params["cyc_rate"]:.2f}', f'{r.pvalues["cyc_rate"]:.1e}',
                    f'{r.params["log1p_ntot"]:.2f}', f'{r.params["log1p_h2h"]:.2f}', f'{r.params["women"]:.2f}'))
    table('S2  Conflict rate on cyclic vs acyclic matchups (directed matches: >=1 prior H2H, strict majority, WElo != 0.5)',
          ['Split', 'Variant', 'H2H depth', 'Cyclic N', 'P(cf|cyc)', 'Acyclic N', 'P(cf|acyc)', 'z', 'p'], link)
    table('S2  Share of conflicts on cyclic matchups', ['Split', 'Variant', 'Cyclic conflicts', 'Conflicts', 'P(cyclic|conflict)'], share)
    table('S2  Conflict logistic on cyclic rate (measurable directed matches)',
          ['Split', 'N', 'cyc_rate', 'p', 'log1p ntot', 'log1p h2h', 'women'], lgt)

    s3 = []
    for split in ['val', 'test']:
        df = data[split]
        cf, directed = masks(df)
        lead = np.sign(df['h2h_w_A'].values - df['h2h_w_B'].values)
        # (a) incremental info: lead indicator beyond logit(WElo), directed matches
        X = sm.add_constant(pd.DataFrame({'lw': logit(df[ELO].values[directed]), 'lead': lead[directed].astype(float)}))
        r = sm.Logit(df['Awin'].values[directed].astype(float), X).fit(disp=0)
        # (b) conflict calibration: leader realised vs WElo-implied win rate
        pl = np.where(lead[cf] > 0, df[ELO].values[cf], 1 - df[ELO].values[cf])        # WElo-implied leader prob
        yl = np.where(lead[cf] > 0, df['Awin'].values[cf], 1 - df['Awin'].values[cf])  # leader won
        # (c) who sides with the H2H leader (= predicts leader to win)?
        sides = {nm: (np.sign(df[c].values - 0.5) == lead) for nm, c in [('GNN', GNN), ('Pin', PIN), ('WElo', ELO)]}
        s3.append((split.upper(), f'{directed.sum():,}', f'{r.params["lead"]:.4f}', f'{r.pvalues["lead"]:.3f}', f'{cf.sum():,}',
                   f'{yl.mean():.3f}', f'{pl.mean():.3f}', *(f'{sides[k][cf].mean():.3f}' for k in ['GNN', 'Pin']),
                   *(f'{sides[k][directed].mean():.3f}' for k in ['GNN', 'Pin', 'WElo'])))
    table('S3  H2H-leader signal on conflicts (sides = share predicting the H2H leader)',
          ['Split', 'Directed N', 'Lead coef', 'p', 'Conflicts', 'Leader won', 'WElo implied',
           'GNN sides (cf)', 'Pin sides (cf)', 'GNN sides (dir)', 'Pin sides (dir)', 'WElo sides (dir)'], s3)

    cf, _ = masks(t)
    slice_rows('S4  Encompassing gain across the structural partition (paper scheme)', t, v, [
        ('no common opponents', t['n_common_opponents__h1'].values == 0),
        ('triad-unmeasurable', t['n_triads_total__h1'].values == 0),
        ('common opponents, no triad', (t['n_common_opponents__h1'].values > 0) & (t['n_triads_total__h1'].values == 0)),  # Table gap_eff row 2
        ('measurable, T=0', (t['n_triads_total__h1'].values > 0) & (t['n_triads_intransitive__h1'].values == 0)),
        ('T=0 (complement)', t['n_triads_intransitive__h1'].values == 0),  # Section 6.2: T_1=0 p
        ('T>0 (published split)', t['n_triads_intransitive__h1'].values > 0),
        ('focal conflict', cf),
    ])


if __name__ == '__main__':
    main()
