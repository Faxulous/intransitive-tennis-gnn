#!/usr/bin/env python3
"""Design-effect-equivalent game evidence under a clustered-binomial approximation.

Treating matches as clustered binary game totals, a pair-stratified one-way ANOVA
moment estimator over pairs with >=2 meetings estimates rho. The design-effect
approximation DEFF = 1 + (m_bar - 1) rho gives a variance-equivalent summary,
n_eff = m_bar / DEFF; it is not a literal count of independent games. Pair-cluster
bootstrap intervals quantify estimator uncertainty. A same-surface variant is
reported for comparison.

Run: python eval/intr/game_icc.py
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import betabinom
from rich.console import Console
from rich.table import Table

con = Console(width=150)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIST = {'ATP': 'data/atp_cleaned.csv', 'WTA': 'data/wta_cleaned.csv'}
OUT_CSV = 'outputs/game_icc.csv'
SEED, B = 8, 1000


def parts(df):  # pair-level ANOVA sufficient statistics
    out = []
    for _, g in df.groupby('pair', sort=False):
        if len(g) < 2:
            continue
        y, m = g['y'].values.astype(float), g['m'].values.astype(float)
        M, p_hat = m.sum(), y.sum() / m.sum()
        p_k = y / m
        out.append(((m * (p_k - p_hat) ** 2).sum(), (m * p_k * (1 - p_k)).sum(),
                    len(g) - 1, M - len(g), M - (m ** 2).sum() / M, len(g)))
    if not out:
        raise ValueError('ICC requires at least one player pair with two meetings')
    return np.asarray(out, dtype=float)


def icc(p):
    ssb, ssw, df_b, df_w, m0_num, _ = p.sum(axis=0)
    msb, msw, m0 = ssb / df_b, ssw / df_w, m0_num / df_b
    rho = (msb - msw) / (msb + (m0 - 1) * msw)
    return rho, m0, int(df_b)


def flip(gw, m, rho):  # P(a Beta-Binomial game redraw reverses the winner); mirrors record_stability.flip_prob
    g = gw / m
    a, b = g * (1 - rho) / rho, (1 - g) * (1 - rho) / rho
    f = betabinom.cdf((m - 1) // 2, m, a, b)
    return f + (0.5 * betabinom.pmf(m // 2, m, a, b) if m % 2 == 0 else 0.0)


def bo3_scores(df):  # winner-perspective set-score strings for completed best-of-three matches
    bo3 = df[df['Best of'] == 3].dropna(subset=['A1', 'B1', 'A2', 'B2'])

    def score(r):
        w, s = r['Awin'] == 1, []
        for i in '123':
            a, b = r[f'A{i}'], r[f'B{i}']
            if pd.isna(a) or pd.isna(b) or (a == 0 and b == 0):
                break
            s.append(f"{int(a if w else b)}-{int(b if w else a)}")
        return ' '.join(s)
    return bo3.apply(score, axis=1).value_counts()


def main():
    rng = np.random.default_rng(SEED)
    rows, sc = [], {}
    for tour, path in HIST.items():
        df = pd.read_csv(path, usecols=['Aname', 'Bname', 'Surface', 'A_games', 'B_games', 'Awin', 'Best of',
                                        'A1', 'B1', 'A2', 'B2', 'A3', 'B3'], low_memory=False)
        df = df.dropna(subset=['A_games', 'B_games'])
        df = df[(df['A_games'] + df['B_games']) > 0]
        sc[tour] = bo3_scores(df)
        # Canonical pair: alphabetical; y = games won by the alphabetically-first player
        first = df['Aname'] < df['Bname']
        df['pair'] = np.where(first, df['Aname'] + '|' + df['Bname'], df['Bname'] + '|' + df['Aname'])
        df['y'] = np.where(first, df['A_games'], df['B_games'])
        df['m'] = df['A_games'] + df['B_games']
        m_bar = df['m'].mean()
        for scope, d in [('all matches', df),
                         ('same surface', df.assign(pair=df['pair'] + '|' + df['Surface']))]:
            p = parts(d)
            rho, m0, dfb = icc(p)
            boot = np.array([icc(p[rng.integers(len(p), size=len(p))])[0] for _ in range(B)])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            deff = 1 + (m_bar - 1) * rho
            rows.append(dict(tour=tour, scope=scope, n_matches=len(d), n_repeat_pairs=len(p),
                             n_repeat_matches=int(p[:, 5].sum()), rho=round(rho, 4),
                             rho_lo=round(lo, 4), rho_hi=round(hi, 4),
                             m_bar=round(m_bar, 1), deff=round(deff, 2),
                             n_eff=round(m_bar / deff, 1),
                             n_eff_m12=round(12 / (1 + 11 * rho), 1),  # 12-game "double bagel" illustration
                             df_between=dfb))
    out = pd.DataFrame(rows)

    # Redraw-severity illustrations (App E): 13-11 closest match + per-tour modal Bo3 scoreline
    am = out['scope'].eq('all matches')
    for i in out.index[am]:
        modal = sc[out.at[i, 'tour']].idxmax()
        gw = sum(int(s.split('-')[0]) for s in modal.split())
        m = gw + sum(int(s.split('-')[1]) for s in modal.split())
        out.at[i, 'modal_bo3'] = modal
        out.at[i, 'f_modal'] = round(flip(gw, m, out.at[i, 'rho']), 3)
        out.at[i, 'f_1311'] = round(flip(13, 24, out.at[i, 'rho']), 3)

    tbl = Table(title="Clustered-binomial moment estimate -> design-effect-equivalent games")
    for c in ['tour', 'scope', 'repeat pairs', 'repeat matches', 'rho', '95% interval',
              'mean games/match', 'DEFF', 'n_eff/match', 'n_eff @ 12 games']:
        tbl.add_column(c, justify='right')
    for r in out.itertuples():
        tbl.add_row(r.tour, r.scope, f'{r.n_repeat_pairs:,}', f'{r.n_repeat_matches:,}',
                    f'{r.rho:.4f}', f'[{r.rho_lo:.4f}, {r.rho_hi:.4f}]', f'{r.m_bar:.1f}',
                    f'{r.deff:.2f}', f'{r.n_eff:.1f}', f'{r.n_eff_m12:.1f}')
    con.print(tbl)
    con.print("[dim]Moment estimates and design-effect equivalents rely on the exchangeable "
              "clustered-binomial approximation; heterogeneity may also contribute to rho.[/dim]")
    tbl = Table(title="Winner-flip probability of a game-level replay")
    for c in ['tour', '13-11 match', 'modal Bo3 scoreline', 'modal Bo3']:
        tbl.add_column(c, justify='right')
    for r in out[am].itertuples():
        tbl.add_row(r.tour, f'{r.f_1311:.3f}', r.modal_bo3, f'{r.f_modal:.3f}')
    con.print(tbl)

    os.makedirs('outputs', exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    con.print(f"\n  saved -> {OUT_CSV}")


if __name__ == '__main__':
    main()
