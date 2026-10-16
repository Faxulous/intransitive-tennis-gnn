#!/usr/bin/env python3
"""Model-difference analyses: where the GNN and WElo predictions actually differ.

TWO analyses in one script:

A. Correctness confusion matrix. For each match: did GNN get the winner right?
   Did WElo? Cross-tabulate. Key cells:
     - GNN-right & WElo-wrong: GNN's unique contribution
     - WElo-right & GNN-wrong: cost of using GNN alone
   McNemar's test on (b,c); if ~equal, models genuinely catch different things
   even at parity accuracy. Also characterise the GNN-only-right subset by
   graph structure (mean IA, triads, common opps) to locate GNN's edge.

B. Murphy's Brier decomposition: B = Reliability - Resolution + Uncertainty.
   Two models at parity Brier can have structurally different reliability/
   resolution profiles - if so, they encode genuinely different signals even
   when their total error looks identical.

Uses 10 quantile bins for Murphy (handles sparse extreme regions).
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.stats import chi2, binomtest

con = Console(width=140)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
RAW_IA     = 'data/predictions/ia/raw_ia.csv'
GNN, ELO   = 'model_prob_non_isolated', 'elo_weighted_prob'
N_BINS     = 10   # Murphy decomposition
SEED       = 8


def load_and_merge():
    df = pd.read_csv(TEST_PREDS, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    raw = pd.read_csv(RAW_IA, parse_dates=['Date'])
    extra = ['raw_intransitivity_measure', 'n_common_opponents',
             'n_triads_total', 'n_triads_intransitive']
    df = df.merge(raw[['Date','Aname','Bname','gender_source'] + extra],
                  on=['Date','Aname','Bname','gender_source'], how='left')
    return df.dropna(subset=[GNN, ELO, 'Awin'] + extra).copy()


def mcnemar(b, c):
    """b = #(GNN right, WElo wrong), c = #(WElo right, GNN wrong).
    Returns (stat, p). Exact binomial if b+c<=25, else chi-sq w/ continuity."""
    n = b + c
    if n == 0: return 0.0, 1.0
    if n <= 25:
        return float(n), binomtest(min(b, c), n, 0.5, alternative='two-sided').pvalue
    stat = (abs(b - c) - 1)**2 / n
    return stat, 1 - chi2.cdf(stat, df=1)


def logloss(p, y, eps=1e-15):  # proper scoring rule, unbounded - tail-sensitive
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def murphy(p, y, n_bins=N_BINS):
    """Bias-corrected Murphy decomposition via quantile bins. Returns (Brier, Rel, Res, Unc),
    B = Rel - Res + Unc. Rel/Res/Unc are the Ferro & Fricker (2012) bias-corrected estimates
    (eq. 11-13): the naive binned estimator overstates reliability (Broecker 2011), badly so
    when reliability is near zero. Corrections cancel in the sum, so B is preserved exactly."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))  # collapse dup edges (rare with continuous p)
    bins = np.digitize(p, edges[1:-1])

    o_bar, n = y.mean(), len(y)
    rel, res, S = 0.0, 0.0, 0.0           # S = Ferro-Fricker correction sum
    for k in np.unique(bins):
        mask = bins == k
        nk = mask.sum()
        if nk == 0: continue
        pk, ok = p[mask].mean(), y[mask].mean()
        rel += nk * (pk - ok)**2
        res += nk * (ok - o_bar)**2
        if nk > 1: S += nk / (nk - 1) * ok * (1 - ok)
    rel, res, S = rel / n, res / n, S / n
    corr = o_bar * (1 - o_bar) / (n - 1)
    rb, sb = rel - S, res - S + corr      # bias-corrected reliability, resolution
    unc = o_bar * (1 - o_bar) + corr      # bias-corrected uncertainty (exactly unbiased)
    rel = max(rb, rb - sb, 0.0)           # Ferro-Fricker flooring (no-op when both positive)
    res = max(sb, sb - rb, 0.0)
    brier = ((p - y)**2).mean()
    return brier, rel, res, unc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
np.random.seed(SEED)
df = load_and_merge()
df['gnn_right'] = ((df[GNN] > 0.5).astype(int) == df['Awin']).astype(int)
df['elo_right'] = ((df[ELO] > 0.5).astype(int) == df['Awin']).astype(int)
con.print(f"Loaded TEST={len(df):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. CORRECTNESS CONFUSION MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENDERS = ['men', 'women', 'pooled']
pct = lambda k, n: f"{k:,} ({k/n*100:.1f}%)"
mc = Table(title="Accuracy and McNemar test (b = GNN right only, c = WElo right only)")
for c in ["Gender", "N", "GNN acc", "WElo acc", "Diff", "Disagree %", "b", "c", "McNemar stat", "p"]:
    mc.add_column(c, justify="right")
for gender in GENDERS:
    sub = df if gender == 'pooled' else df[df['gender_source'] == gender]
    n = len(sub)
    both_r  = ((sub['gnn_right']==1) & (sub['elo_right']==1)).sum()
    both_w  = ((sub['gnn_right']==0) & (sub['elo_right']==0)).sum()
    b_gnn_only = ((sub['gnn_right']==1) & (sub['elo_right']==0)).sum()   # McNemar b
    c_elo_only = ((sub['gnn_right']==0) & (sub['elo_right']==1)).sum()   # McNemar c
    stat, p = mcnemar(b_gnn_only, c_elo_only)

    tbl = Table(title=f"Correctness confusion: {gender} (N={n:,})")
    for c in ["", "WElo right", "WElo wrong", "Row total"]:
        tbl.add_column(c, justify="right")
    tbl.add_row("GNN right", pct(both_r, n), pct(b_gnn_only, n), f"{both_r + b_gnn_only:,}")
    tbl.add_row("GNN wrong", pct(c_elo_only, n), pct(both_w, n), f"{both_w + c_elo_only:,}")
    tbl.add_row("Col total", f"{both_r + c_elo_only:,}", f"{b_gnn_only + both_w:,}", f"{n:,}")
    con.print(tbl)

    gnn_acc, elo_acc = (both_r + b_gnn_only) / n, (both_r + c_elo_only) / n
    mc.add_row(gender, f"{n:,}", f"{gnn_acc:.3f}", f"{elo_acc:.3f}", f"{gnn_acc-elo_acc:.3f}",
               f"{(b_gnn_only + c_elo_only)/n*100:.1f}", f"{b_gnn_only:,}", f"{c_elo_only:,}", f"{stat:.2f}", f"{p:.4f}")
con.print(mc)

# Characterize the GNN-only-right cell vs WElo-only-right cell: graph structure
cells = [
    ('Both right',    (df['gnn_right']==1) & (df['elo_right']==1)),
    ('GNN right only',(df['gnn_right']==1) & (df['elo_right']==0)),
    ('WElo right only',(df['gnn_right']==0) & (df['elo_right']==1)),
    ('Both wrong',    (df['gnn_right']==0) & (df['elo_right']==0)),
]
tbl = Table(title="Mean graph-structure features by correctness cell (pooled)")
for c in ["Cell", "N", "% of data", "Raw IA", "Common opps", "Total triads", "Intransitive triads", "% IA>0"]:
    tbl.add_column(c, justify="right", no_wrap=c == "Cell")
for name, mask in cells:
    s = df[mask]
    tbl.add_row(name, f"{len(s):,}", f"{len(s)/len(df)*100:.1f}",
                f"{s['raw_intransitivity_measure'].mean():.4f}",
                f"{s['n_common_opponents'].mean():.1f}",
                f"{s['n_triads_total'].mean():.1f}",
                f"{s['n_triads_intransitive'].mean():.2f}",
                f"{(s['raw_intransitivity_measure'] > 0).mean()*100:.1f}")
con.print(tbl)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. MURPHY BRIER DECOMPOSITION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tbl = Table(title="Bias-corrected Murphy decomposition, B = Rel - Res + Unc")
for c in ["Gender", "Model", "N", "Brier", "LogLoss", "Reliability", "Resolution", "Uncertainty", "Res - Rel"]:
    tbl.add_column(c, justify="right")
dif = Table(title="Differences, GNN - WElo")
for c in ["Gender", "Brier", "LogLoss", "Reliability", "Resolution"]:
    dif.add_column(c, justify="right")
for gender in GENDERS:
    sub = df if gender == 'pooled' else df[df['gender_source'] == gender]
    y, m = sub['Awin'].values, {}
    for mname, col in [('GNN', GNN), ('WElo', ELO)]:
        p = sub[col].values
        m[mname] = (*murphy(p, y, n_bins=N_BINS), logloss(p, y))  # (Brier, Rel, Res, Unc, LogLoss)
        b, rel, res, unc, ll = m[mname]
        tbl.add_row(gender if mname == 'GNN' else '', mname, f"{len(sub):,}", f"{b:.5f}", f"{ll:.5f}", f"{rel:.5f}",
                    f"{res:.5f}", f"{unc:.5f}", f"{res - rel:.5f}", end_section=mname == 'WElo')
    (bg, rg, sg, _, lg), (be, re_, se, _, le) = m['GNN'], m['WElo']
    dif.add_row(gender, f"{bg-be:.5f}", f"{lg-le:.5f}", f"{rg-re_:.5f}", f"{sg-se:.5f}")
con.print(tbl)
con.print(dif)

slices = [('pooled', np.ones(len(df), bool)),
          ('IA=0',   (df['raw_intransitivity_measure'] == 0).values),
          ('IA>0',   (df['raw_intransitivity_measure'] >  0).values)]
tbl = Table(title="Brier, log loss and Murphy components by raw-triad IA slice (pooled; dB, dLL = GNN - WElo)")
for c in ["Slice", "N", "GNN Brier", "WElo Brier", "dB", "GNN LogLoss", "WElo LogLoss", "dLL",
          "GNN Rel", "WElo Rel", "GNN Res", "WElo Res"]:
    tbl.add_column(c, justify="right")
for sname, mask in slices:
    sub = df[mask]; y = sub['Awin'].values
    pg, pe = sub[GNN].values, sub[ELO].values
    bg, rg, sg, _ = murphy(pg, y); be, re_, se, _ = murphy(pe, y)
    llg, lle = logloss(pg, y), logloss(pe, y)
    tbl.add_row(sname, f"{len(sub):,}",
                f"{bg:.5f}", f"{be:.5f}", f"{bg-be:.5f}",
                f"{llg:.5f}", f"{lle:.5f}", f"{llg-lle:.5f}",
                f"{rg:.5f}", f"{re_:.5f}", f"{sg:.5f}", f"{se:.5f}")
con.print(tbl)
