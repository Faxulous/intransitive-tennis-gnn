#!/usr/bin/env python3
"""Honest record-uncertainty stability of triad labels and the encompassing gain.

A match-only bootstrap (Binomial(n, w/n) per edge) can never flip a unanimous record
(w in {0, n}, including every single-meeting edge), so its retention rates are
mechanically inflated exactly where evidence is thinnest. This script propagates
GAME-level evidence instead. Each historical meeting j gets a
flip probability f_j = P(a game-level replay of the meeting reverses its winner),
with the replay drawn Beta-Binomial(m_j games, winner share, rho_tour) - the same
intra-match correlation rho reported in Section 6.3 (outputs/game_icc.csv). Three
analyses share one causal streaming pass (triad rules identical to
data/scripts/raw_intransitivity.py, baselines validated exactly against the production raw_ia*.csv):

A TWO-STAGE bootstrap: resample each edge's n meetings with replacement, then
  redraw each drawn meeting's result from its game record. Collapses EXACTLY to
  W ~ Binomial(n, q), q = mean_j P(drawn meeting j contributes a win). The
  match-only variant (f=0, i.e. q = w/n) is reported alongside for comparison.
B Exact leave-one-meeting-out: every single deletion of a contributing meeting,
  grouped into per-edge win/loss deletion classes; label survival + destinations.
D GLOBAL game-redraw worlds: B alternative histories, each historical match's
  winner redrawn ONCE (P(flip) = f_k) and used consistently across all queries;
  labels AND the encompassing dB / eta / DM p (fixed VAL scheme, mirroring
  threshold_sens.py) recomputed per world -> distribution of the headline result.

Run: python eval/intr/record_stability.py [--quick]
"""
import os, sys, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import betabinom, norm
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table

con = Console(width=170)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.getcwd())
from data.scripts.raw_intransitivity import weak_pat

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
B = 1000
HS = [1, 2, 3, 4]
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
VAL_PREDS = 'data/predictions/preds_w_ia_val.csv'
RAW = {1: 'data/predictions/ia/raw_ia.csv', 2: 'data/predictions/ia/raw_ia_h2.csv',
       3: 'data/predictions/ia/raw_ia_h3.csv', 4: 'data/predictions/ia/raw_ia_h4.csv'}
HIST = {'men': 'data/atp_cleaned.csv', 'women': 'data/wta_cleaned.csv'}
ICC_CSV = 'outputs/game_icc.csv'                    # rho per tour (scope='all matches')
GNN, ELO, PIN = 'model_prob_non_isolated', 'elo_weighted_prob', 'PSA_ip'
EPS = 1e-6
OUT = 'outputs/record_stability'
QUICK_Q, QUICK_B = 600, 200                          # --quick: queries/gender, replicates


def flip_prob(gw, m, rho):  # P(game-level replay reverses the winner), vectorised
    """gw: winner's games, m: total games. Replay X ~ BetaBinom(m, g, rho); winner
    loses the replay if X < m/2, ties split 0.5. Requires rho > 0 (measured ICCs are)."""
    g = gw / m
    a, b = g * (1 - rho) / rho, (1 - g) * (1 - rho) / rho
    k = (m - 1) // 2                                 # largest count strictly below m/2
    with np.errstate(invalid='ignore'):
        f = betabinom.cdf(k, m, np.maximum(a, 1e-12), np.maximum(b, 1e-12))
        tie = np.where(m % 2 == 0, betabinom.pmf(m // 2, m, np.maximum(a, 1e-12), np.maximum(b, 1e-12)), 0.0)
    f = f + 0.5 * tie
    f = np.where(g >= 1.0, 0.0, np.where(g <= 0.0, 1.0, f))   # degenerate shares: exact
    return f


def signs(W, n):  # majority sign of W wins out of n (0 = tie)
    return np.sign(2 * W - n)


def cyc_pat(sAB, sAC, sBC):  # 3-cycle pattern; sAB may broadcast
    return ((sAB == 1) & (sBC == 1) & (sAC == -1)) | ((sAB == -1) & (sBC == -1) & (sAC == 1))


def measurable_pat(sAB, sAC, sBC):
    """Relations classifiable as a strict cycle or a consistent weak order."""
    return cyc_pat(sAB, sAC, sBC) | weak_pat(sAB, sAC, sBC)


def lomo_query(h, wAB, nAB, rec, nAC, nBC, gate, sAB0, sAC0, sBC0, ok0, cyc0):
    """Exact single-deletion classes for one query at threshold h.
    Returns (total_deletions, flip_mult, to_other_mult, to_unmeas_mult, base_code)."""
    ntot0, nint0 = int(ok0.sum()), int((ok0 & cyc0).sum())
    base = 2 if nint0 > 0 else (1 if ntot0 > 0 else 0)
    if base == 0:
        return 0, 0, 0, 0, 0
    tot_del = int(nAB + (nAC[gate] + nBC[gate]).sum())        # AB + edges of gated triads
    flip = to_other = to_unm = 0

    def acc(mult, new):  # accumulate a deletion class of size mult with outcome new
        nonlocal flip, to_other, to_unm
        if new != base:
            flip += mult
            if new == 0:
                to_unm += mult
            else:
                to_other += mult

    # AB deletions: gates on AC/BC unchanged; sAB may change or nAB drop below h
    for dw, mult in [(1, int(wAB)), (0, int(nAB - wAB))]:
        if mult == 0:
            continue
        nn = nAB - 1
        if nn < h:
            acc(mult, 0)
            continue
        sABn = int(signs(wAB - dw, nn))
        okn = gate & measurable_pat(sABn, sAC0, sBC0)
        nintn = int((okn & cyc_pat(sABn, sAC0, sBC0)).sum())
        acc(mult, 2 if nintn > 0 else (1 if okn.any() else 0))

    # AC / BC deletions: only gated triads can change; one triad affected per class
    idx = np.flatnonzero(gate)
    for side, w_e, n_e, s_e in [(0, rec[:, 0], nAC, sAC0), (1, rec[:, 2], nBC, sBC0)]:
        other_ok = (nBC >= h) if side == 0 else (nAC >= h)    # unchanged gate half
        s_oth = sBC0 if side == 0 else sAC0                   # unchanged sign of same triad
        for dw in (1, 0):
            mult_e = w_e if dw == 1 else n_e - w_e
            for c in idx:
                mult = int(mult_e[c])
                if mult == 0:
                    continue
                nn = int(n_e[c]) - 1
                if nn < h:                                     # triad drops out entirely
                    okn = cn = 0
                else:
                    sn = int(signs(int(w_e[c]) - dw, nn))
                    sac, sbc = (sn, s_oth[c]) if side == 0 else (s_oth[c], sn)
                    okn = int(other_ok[c] and measurable_pat(sAB0, sac, sbc))
                    cn = int(okn and cyc_pat(sAB0, sac, sbc))
                nintn = nint0 - int(ok0[c] & cyc0[c]) + cn
                ntotn = ntot0 - int(ok0[c]) + okn
                acc(mult, 2 if nintn > 0 else (1 if ntotn > 0 else 0))
    return tot_del, flip, to_other, to_unm, base


def stream(gender, rho, rng, quick):
    """One causal pass: baselines, two-stage bootstrap (A), LOMO (B), world labels (D)."""
    hist = pd.read_csv(HIST[gender], usecols=['Date', 'Aname', 'Bname', 'Awin', 'A_games', 'B_games'], low_memory=False)
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist.sort_values('Date', kind='mergesort').reset_index(drop=True)
    gw = np.where(hist['Awin'] == 1, hist['A_games'], hist['B_games']).astype(float)
    m = (hist['A_games'] + hist['B_games']).values.astype(float)
    f_all = flip_prob(gw, m, rho)
    Bn = QUICK_B if quick else B
    keep = rng.random((len(hist), Bn)) >= f_all[:, None]      # world k keeps observed winner
    hd, hA, hB = hist['Date'].values, hist['Aname'].values, hist['Bname'].values
    hW = hist['Awin'].astype(np.int8).values

    preds = pd.read_csv(TEST_PREDS, usecols=['Date', 'Aname', 'Bname', 'gender_source'], low_memory=False)
    q = preds[preds['gender_source'] == gender].copy()
    q['Date'] = pd.to_datetime(q['Date'])
    q = q.sort_values('Date', kind='mergesort')
    if quick:
        q = q.head(QUICK_Q)
    con.print(f"  [bold]{gender}[/bold]: {len(q):,} queries vs {len(hist):,} historical matches "
              f"(mean f {f_all.mean():.3f}, median {np.median(f_all):.3f}, f>0.25: {(f_all > .25).mean():.1%})")

    wins = defaultdict(lambda: defaultdict(int))              # observed w[a][b]
    wwins = defaultdict(lambda: defaultdict(lambda: np.zeros(Bn, np.int16)))  # world wins
    meets = defaultdict(list)                                 # (u<v) -> [(win_is_u, f_j)]
    played = defaultdict(set)
    h_i, rows, codes = 0, [], {h: [] for h in HS}
    for qi, (t, A, Bp) in enumerate(zip(q['Date'].values, q['Aname'].values, q['Bname'].values)):
        if qi and qi % 2000 == 0:
            con.print(f"    {qi:,}/{len(q):,} queries")
        while h_i < len(hist) and hd[h_i] < t:
            a, b, w = hA[h_i], hB[h_i], int(hW[h_i])
            wins[a][b] += w; wins[b][a] += 1 - w
            kw = keep[h_i].astype(np.int16)
            win, lose = (a, b) if w else (b, a)
            wwins[win][lose] += kw; wwins[lose][win] += (1 - kw)
            u, v = (a, b) if a < b else (b, a)
            meets[(u, v)].append(((a == u) == bool(w), f_all[h_i]))
            played[a].add(b); played[b].add(a)
            h_i += 1
        common = sorted((played[A] & played[Bp]) - {A, Bp})
        wAB, wBA = wins[A][Bp], wins[Bp][A]
        nAB = wAB + wBA
        rec = np.array([[wins[A][C], wins[C][A], wins[Bp][C], wins[C][Bp]] for C in common],
                       dtype=np.int64).reshape(-1, 4)
        nAC, nBC = rec[:, 0] + rec[:, 1], rec[:, 2] + rec[:, 3]
        sAC0, sBC0 = signs(rec[:, 0], nAC), signs(rec[:, 2], nBC)
        sAB0 = int(signs(wAB, nAB)) if nAB > 0 else 0

        def q_of(X, Y, wXY, nXY):  # two-stage collapse: mean P(drawn meeting is a win for X)
            u, v = (X, Y) if X < Y else (Y, X)
            lst = meets[(u, v)]
            assert len(lst) == nXY and sum((wu == (X == u)) for wu, _ in lst) == wXY, (X, Y)
            return sum((1 - fj) if (wu == (X == u)) else fj for wu, fj in lst) / nXY

        row = dict(Date=t, Aname=A, Bname=Bp, gender_source=gender)
        if len(rec) and nAB > 0:
            qAB = q_of(A, Bp, wAB, nAB)
            qAC = np.array([q_of(A, C, rec[k, 0], nAC[k]) for k, C in enumerate(common)])
            qBC = np.array([q_of(Bp, C, rec[k, 2], nBC[k]) for k, C in enumerate(common)])
            draws = {}
            for tag, pAB, pAC, pBC in [('m', wAB / nAB, rec[:, 0] / nAC, rec[:, 2] / nBC),
                                       ('g', qAB, qAC, qBC)]:
                sab = signs(rng.binomial(nAB, pAB, size=Bn), nAB)
                sac = signs(rng.binomial(nAC[:, None], pAC[:, None], size=(len(rec), Bn)), nAC[:, None])
                sbc = signs(rng.binomial(nBC[:, None], pBC[:, None], size=(len(rec), Bn)), nBC[:, None])
                draws[tag] = (sab, sac, sbc, cyc_pat(sab, sac, sbc))
            wD_AB = wwins[A][Bp]
            wD_AC = np.stack([wwins[A][C] for C in common])
            wD_BC = np.stack([wwins[Bp][C] for C in common])
            sabD = signs(wD_AB, nAB)
            sacD, sbcD = signs(wD_AC, nAC[:, None]), signs(wD_BC, nBC[:, None])
            cycD = cyc_pat(sabD, sacD, sbcD)
        for h in HS:
            gate = (nAC >= h) & (nBC >= h) if len(rec) else np.zeros(0, bool)
            if nAB < h or not gate.any():
                row.update({f'ntot_h{h}': 0, f'nint_h{h}': 0})
                codes[h].append(np.zeros(Bn, np.int8))
                continue
            ok0 = gate & measurable_pat(sAB0, sAC0, sBC0)
            cyc0 = cyc_pat(sAB0, sAC0, sBC0) & ok0
            row[f'ntot_h{h}'] = int(ok0.sum())
            row[f'nint_h{h}'] = int(cyc0.sum())
            for tag in ('m', 'g'):                            # A: marginal retention per variant
                sab, sac, sbc, cyc = draws[tag]
                ok = gate[:, None] & measurable_pat(sab, sac, sbc)
                nint_b, ntot_b = (ok & cyc).sum(0), ok.sum(0)
                row[f'p_int_{tag}_h{h}'] = float((nint_b > 0).mean())
                row[f'p_trans_{tag}_h{h}'] = float(((ntot_b > 0) & (nint_b == 0)).mean())
            okD = gate[:, None] & measurable_pat(sabD, sacD, sbcD)
            nintD, ntotD = (okD & cycD).sum(0), okD.sum(0)
            codes[h].append(np.where(nintD > 0, 2, np.where(ntotD > 0, 1, 0)).astype(np.int8))
            td, fl, oth, unm, base = lomo_query(h, wAB, nAB, rec, nAC, nBC, gate, sAB0, sAC0, sBC0, ok0, cyc0)
            row.update({f'lomo_del_h{h}': td, f'lomo_flip_h{h}': fl,
                        f'lomo_other_h{h}': oth, f'lomo_unm_h{h}': unm})
        rows.append(row)
    return pd.DataFrame(rows), {h: np.vstack(codes[h]) for h in HS}


def encompassing(out, codes):
    """Fixed-scheme dB/eta/DM per world (mirrors threshold_sens.py exactly)."""
    logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    val = pd.read_csv(VAL_PREDS, low_memory=False).dropna(subset=[GNN, ELO, PIN, 'Awin'])
    yv = val['Awin'].values
    Xv = np.column_stack([logit(val[ELO]), logit(val[GNN])])
    test = pd.read_csv(TEST_PREDS, low_memory=False)
    test['Date'] = pd.to_datetime(test['Date'])
    mg = out.merge(test, on=['Date', 'Aname', 'Bname', 'gender_source'], how='left', validate='1:1')
    if len(mg) != len(out):
        raise RuntimeError("preds merge changed row count")
    ok = mg[[GNN, ELO, PIN, 'Awin']].notna().all(1).values
    y = mg['Awin'].values[ok]
    Xt = np.column_stack([logit(mg[ELO]), logit(mg[GNN])])[ok]
    lr = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs')
    p_wg = lr.fit(Xv, yv).predict_proba(Xt)[:, 1]
    p_w = lr.fit(Xv[:, :1], yv).predict_proba(Xt[:, :1])[:, 1]
    d = (p_w - y) ** 2 - (p_wg - y) ** 2                      # per-match gain (positive = W+G better)
    d_pin = (p_w - y) ** 2 - (mg[PIN].values[ok] - y) ** 2    # headroom to Pinnacle
    res, worlds = [], {}
    for h in HS:
        C = codes[h][ok]
        base_int = (mg[f'nint_h{h}'].values[ok] > 0)
        base_trn = (mg[f'ntot_h{h}'].values[ok] > 0) & ~base_int
        for cell, bmask, cc in [('intransitive', base_int, 2), ('transitive', base_trn, 1)]:
            dB0 = d[bmask].mean()
            eta0 = dB0 / d_pin[bmask].mean() * 100
            sd0 = d[bmask].std(ddof=1) / np.sqrt(bmask.sum())
            p0 = 1 - norm.cdf(dB0 / sd0)
            M = (C == cc).astype(np.float32)                  # (nOk, B) world membership
            cnt = M.sum(0)
            good = cnt >= 10
            S1, S2, Sp = d @ M, (d * d) @ M, d_pin @ M
            dBb = np.where(good, S1 / np.maximum(cnt, 1), np.nan)
            varb = (S2 / np.maximum(cnt, 1) - dBb ** 2) * cnt / np.maximum(cnt - 1, 1)
            pb = 1 - norm.cdf(dBb / np.sqrt(varb / np.maximum(cnt, 1)))
            etab = dBb / (Sp / np.maximum(cnt, 1)) * 100
            g = dBb[good]
            worlds[(h, cell)] = dBb                           # per-world series (nan where cell < 10)
            res.append(dict(h=h, cell=cell, N0=int(bmask.sum()), dB0=dB0, eta0=eta0, p0=p0,
                            n_med=float(np.median(cnt)), dB_med=float(np.median(g)),
                            dB_q05=float(np.quantile(g, .05)), dB_q95=float(np.quantile(g, .95)),
                            pos=float((g > 0).mean()), sig=float((pb[good] < .05).mean()),
                            eta_med=float(np.nanmedian(etab[good]))))
    return pd.DataFrame(res), worlds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help=f'{QUICK_Q} queries/gender, B={QUICK_B}')
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    icc = pd.read_csv(ICC_CSV)
    rho = {g: float(icc[(icc['tour'] == t) & (icc['scope'] == 'all matches')]['rho'].iloc[0])
           for g, t in [('men', 'ATP'), ('women', 'WTA')]}
    con.print(f"Causal pass + bootstrap (B={QUICK_B if args.quick else B}{', QUICK' if args.quick else ''})")
    parts = [stream(g, rho[g], rng, args.quick) for g in ['men', 'women']]
    out = pd.concat([p[0] for p in parts], ignore_index=True)
    codes = {h: np.vstack([p[1][h] for p in parts]) for h in HS}

    con.print("Validate baseline counts against production raw_ia files")
    for h in HS:
        r = pd.read_csv(RAW[h], parse_dates=['Date'])
        mm = out.merge(r[['Date', 'Aname', 'Bname', 'gender_source', 'n_triads_total', 'n_triads_intransitive']],
                       on=['Date', 'Aname', 'Bname', 'gender_source'], validate='1:1')
        if len(mm) != len(out):
            raise RuntimeError(f"h={h}: merge lost rows")
        bad = (mm[f'ntot_h{h}'] != mm['n_triads_total']) | (mm[f'nint_h{h}'] != mm['n_triads_intransitive'])
        if bad.any():
            raise RuntimeError(f"h={h}: {bad.sum()} baseline mismatches vs {RAW[h]}")
        con.print(f"  h={h}: [green]exact match[/green] on {len(mm):,} queries")

    con.print("A/B: label retention (match-only vs two-stage game) + exact LOMO")
    tbl = Table(title="Label stability: frozen match-only bootstrap vs game-evidence propagation")
    for c in ['h', 'cell', 'N', 'keep (match-only)', 'keep (two-stage)', '>=80% (2s)',
              'world keep (corr.)', 'LOMO survive-all', 'LOMO flip/deletion', 'flips-> other:unmeas']:
        tbl.add_column(c, justify='right')
    summ = []
    for h in HS:
        base_int = out[f'nint_h{h}'] > 0
        base_trn = (out[f'ntot_h{h}'] > 0) & ~base_int
        for cell, msk, cc in [('intransitive', base_int, 2), ('transitive', base_trn, 1)]:
            s = out[msk]
            wk = float((codes[h][msk.values] == cc).mean())
            keep_m = s[f'p_int_m_h{h}'] if cell == 'intransitive' else s[f'p_trans_m_h{h}']
            keep_g = s[f'p_int_g_h{h}'] if cell == 'intransitive' else s[f'p_trans_g_h{h}']
            dele, fl = s[f'lomo_del_h{h}'].sum(), s[f'lomo_flip_h{h}'].sum()
            surv = float((s[f'lomo_flip_h{h}'] == 0).mean())
            oth, unm = s[f'lomo_other_h{h}'].sum(), s[f'lomo_unm_h{h}'].sum()
            tbl.add_row(str(h) if cell == 'intransitive' else '', cell, f"{len(s):,}", f"{keep_m.mean():.3f}",
                        f"{keep_g.mean():.3f}", f"{(keep_g >= .8).mean():.1%}", f"{wk:.3f}", f"{surv:.1%}",
                        f"{fl / dele:.3f}", f"{oth / max(fl, 1):.0%}:{unm / max(fl, 1):.0%}", end_section=cell == 'transitive')
            summ.append(dict(h=h, cell=cell, N=len(s), keep_match=keep_m.mean(), keep_2stage=keep_g.mean(),
                             keep80_2stage=(keep_g >= .8).mean(), keep_world=wk, lomo_survive=surv,
                             lomo_flip_per_del=fl / dele, lomo_to_other=oth / max(fl, 1), lomo_to_unmeas=unm / max(fl, 1)))
    con.print(tbl)

    con.print("D: encompassing gain across game-redraw worlds (fixed scheme)")
    enc, worlds = encompassing(out, codes)
    tbl2 = Table(title="dB distribution over worlds (baseline dB0 must match Table threshold_sens)")
    for c in ['h', 'cell', 'N0', 'dB0', 'eta0 %', 'p0', 'N med', 'dB med', 'dB [5,95]%', 'P(dB>0)', 'P(p<.05)', 'eta med %']:
        tbl2.add_column(c, justify='right')
    for r in enc.itertuples():
        tbl2.add_row(str(r.h) if r.cell == 'intransitive' else '', r.cell, f"{r.N0:,}", f"{r.dB0:.4f}", f"{r.eta0:.1f}", f"{r.p0:.3f}",
                     f"{r.n_med:,.0f}", f"{r.dB_med:.4f}", f"[{r.dB_q05:.4f}, {r.dB_q95:.4f}]",
                     f"{r.pos:.1%}", f"{r.sig:.1%}", f"{r.eta_med:.1f}", end_section=r.cell == 'transitive')
    con.print(tbl2)

    tbl3 = Table(title="Per-world contrast: dB(intransitive) - dB(transitive)")
    for c in ['h', 'contrast0', 'median', '[5,95]%', 'P(>0)']:
        tbl3.add_column(c, justify='right')
    for h in HS:
        di, dt = worlds[(h, 'intransitive')], worlds[(h, 'transitive')]
        c0 = float(enc[(enc.h == h) & (enc.cell == 'intransitive')].dB0.iloc[0]
                   - enc[(enc.h == h) & (enc.cell == 'transitive')].dB0.iloc[0])
        dd = di - dt
        dd = dd[~np.isnan(dd)]
        tbl3.add_row(str(h), f"{c0:.4f}", f"{np.median(dd):.4f}",
                     f"[{np.quantile(dd, .05):.4f}, {np.quantile(dd, .95):.4f}]", f"{(dd > 0).mean():.1%}")
    con.print(tbl3)

    os.makedirs('outputs', exist_ok=True)
    sfx = '_quick' if args.quick else ''
    out.to_csv(f'{OUT}_per_match{sfx}.csv', index=False)
    pd.DataFrame(summ).to_csv(f'{OUT}{sfx}.csv', index=False)
    enc.to_csv(f'{OUT}_encompassing{sfx}.csv', index=False)
    np.savez_compressed(f'{OUT}_worlds{sfx}.npz', **{f'{h}_{c}': v for (h, c), v in worlds.items()})
    con.print(f"\n  -> {OUT}{sfx}.csv, {OUT}_encompassing{sfx}.csv, {OUT}_per_match{sfx}.csv")


if __name__ == '__main__':
    main()
