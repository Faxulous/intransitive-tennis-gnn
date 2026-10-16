#!/usr/bin/env python3
"""Model-free raw triad intransitivity for each TEST match (avoids circularity with the model).

For match (A,B,t): count common-opponent triads C such that A,B,C form an
intransitive cycle under head-to-head majority (Date < t only). Output keyed
to preds_w_ia.csv so the eval scripts can merge a raw-IA column.
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table
con = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
ATP_HIST = 'data/atp_cleaned.csv'
WTA_HIST = 'data/wta_cleaned.csv'
ATP_VAL_HIST = 'data/atp_cleaned_VAL.csv'   # full history up to val date (VAL builds)
WTA_VAL_HIST = 'data/wta_cleaned_VAL.csv'
TEST_PREDS = 'data/predictions/preds_w_ia.csv'
VAL_PREDS  = 'data/predictions/preds_w_ia_val.csv'
OUT_TEST = 'data/predictions/ia/raw_ia.csv'
OUT_VAL  = 'data/predictions/ia/raw_ia_val.csv'
REQ_HIST_COLS_MATCH = ['Date', 'Aname', 'Bname', 'Awin']                       # match-level
REQ_HIST_COLS_SETS  = ['Date', 'Aname', 'Bname', 'Awin', 'Asets', 'Bsets']     # set-level
REQ_HIST_COLS_GAMES = ['Date', 'Aname', 'Bname', 'Awin', 'A_games', 'B_games'] # game-level
REQ_PRED_COLS = ['Date', 'Aname', 'Bname', 'gender_source']


def weak_pat(sAB, sAC, sBC):
    """Whether complete pairwise signs admit a transitive weak ordering."""
    bad = ((sAB >= 0) & (sBC >= 0) & (sAC < 0)) | ((sAC >= 0) & (sBC <= 0) & (sAB < 0))
    bad |= ((sAB <= 0) & (sAC >= 0) & (sBC < 0)) | ((sBC >= 0) & (sAC <= 0) & (sAB > 0))
    bad |= ((sAC <= 0) & (sAB >= 0) & (sBC > 0)) | ((sBC <= 0) & (sAB <= 0) & (sAC > 0))
    return np.logical_not(bad)


def compute_raw_ia(pred_df: pd.DataFrame, hist_df: pd.DataFrame,
                   min_h2h: int = 1, granularity: str = 'matches') -> pd.DataFrame:
    """Causal raw-triad IA: one streaming pass over time-ordered (history + queries).

    wins[i][j] = # times i beat j (matches) or # sets i won off j (sets).
    nmatch[i][j] = # prior matches between i and j (granularity-invariant gate).
    played[i]  = set of opponents of i.
    For each query (A,B,t): common = played[A] & played[B] - {A,B}.
    Triad (A,B,C) uses sign(w(A,B)-0.5), sign(w(B,C)-0.5), sign(w(A,C)-0.5),
    where w(i,j) = wins[i][j] / (wins[i][j] + wins[j][i]). Three strict
    directions forming A>B>C>A (or its reverse) are intransitive. Relations that
    admit a consistent weak order, including A=B>C, A>B=C, and A=B=C, are
    transitive. Inconsistent tied relations contribute no triad.
    min_h2h: require each triad edge (A-C, B-C, A-B) to have >= min_h2h prior
    MATCHES (not sets) - keeps the threshold comparable across granularities.
    granularity='sets' upweights dominant matches (3-0 vs 3-2) and breaks ties
    that arise at match level (e.g. 1-1 H2H may become 5-3 in sets)."""
    if granularity not in ('matches', 'sets', 'games'):
        raise ValueError(f"granularity must be 'matches', 'sets' or 'games', got {granularity}")
    REQ_HIST_COLS = {'sets': REQ_HIST_COLS_SETS, 'games': REQ_HIST_COLS_GAMES}.get(granularity, REQ_HIST_COLS_MATCH)
    for c in REQ_HIST_COLS:
        if c not in hist_df.columns: raise ValueError(f"hist missing column: {c}")
    for c in REQ_PRED_COLS:
        if c not in pred_df.columns: raise ValueError(f"pred missing column: {c}")

    # Prep: datetime, stable order (hist first per date so t-equal matches still causal for ties)
    hist = hist_df[REQ_HIST_COLS].copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist.sort_values('Date', kind='mergesort').reset_index(drop=True)
    hist_dates = hist['Date'].values
    hist_A = hist['Aname'].values
    hist_B = hist['Bname'].values
    hist_Aw = hist['Awin'].astype(np.int8).values
    if granularity == 'sets':
        hist_As = hist['Asets'].astype(np.int16).values
        hist_Bs = hist['Bsets'].astype(np.int16).values
    if granularity == 'games':
        hist_Ag = hist['A_games'].fillna(0).astype(np.int16).values
        hist_Bg = hist['B_games'].fillna(0).astype(np.int16).values

    q = pred_df[REQ_PRED_COLS].copy().reset_index().rename(columns={'index': '_qidx'})
    q['Date'] = pd.to_datetime(q['Date'])
    q = q.sort_values('Date', kind='mergesort').reset_index(drop=True)
    q_dates = q['Date'].values
    q_A = q['Aname'].values
    q_B = q['Bname'].values

    wins = defaultdict(lambda: defaultdict(int))  # wins[i][j]: # i beat j (or sets won)
    nmatch = defaultdict(lambda: defaultdict(int)) # nmatch[i][j]: # prior matches (gate)
    played = defaultdict(set)                      # played[i]: opponents of i
    h_i = 0                                        # history cursor
    n_q = len(q)
    out_meas  = np.zeros(n_q, dtype=np.float32)
    out_conf  = np.zeros(n_q, dtype=np.float32)
    out_nco   = np.zeros(n_q, dtype=np.int32)
    out_ntot  = np.zeros(n_q, dtype=np.int32)
    out_nint  = np.zeros(n_q, dtype=np.int32)

    con.print(f"  Processing {n_q:,} queries against {len(hist):,} historical matches...")
    last_pct = -1
    for qi in range(n_q):
        t = q_dates[qi]
        # Absorb all historical matches with Date < t into the state
        while h_i < len(hist) and hist_dates[h_i] < t:
            a, b, aw = hist_A[h_i], hist_B[h_i], hist_Aw[h_i]
            if granularity == 'sets':
                wins[a][b] += int(hist_As[h_i]); wins[b][a] += int(hist_Bs[h_i])
            elif granularity == 'games':
                wins[a][b] += int(hist_Ag[h_i]); wins[b][a] += int(hist_Bg[h_i])
            else:
                if aw == 1:   wins[a][b] += 1
                else:         wins[b][a] += 1
            nmatch[a][b] += 1; nmatch[b][a] += 1   # match count gate, granularity-invariant
            played[a].add(b); played[b].add(a)
            h_i += 1

        A, B = q_A[qi], q_B[qi]
        common = (played[A] & played[B]) - {A, B}
        wAB_plus, wAB_minus = wins[A][B], wins[B][A]
        sAB = 1 if wAB_plus > wAB_minus else (-1 if wAB_plus < wAB_minus else 0)
        m_AB = nmatch[A][B]                             # match-count gate (granularity-invariant)

        n_tri_tot, n_tri_int = 0, 0
        for C in common:
            wAC, wCA = wins[A][C], wins[C][A]
            wBC, wCB = wins[B][C], wins[C][B]
            if nmatch[A][C] < min_h2h or nmatch[B][C] < min_h2h:   # >=min_h2h prior MATCHES on each triad edge
                continue
            if m_AB < min_h2h:                          # also guard the (A,B) edge itself
                continue
            sAC = 1 if wAC > wCA else (-1 if wAC < wCA else 0)
            sBC = 1 if wBC > wCB else (-1 if wBC < wCB else 0)
            cyc = (sAB == 1 and sBC == 1 and sAC == -1) or (sAB == -1 and sBC == -1 and sAC == 1)
            if not (cyc or weak_pat(sAB, sAC, sBC)):    # exclude only inconsistent tied relations
                continue
            n_tri_tot += 1
            n_tri_int += int(cyc)

        out_nco[qi] = len(common)
        out_ntot[qi] = n_tri_tot
        out_nint[qi] = n_tri_int
        out_meas[qi] = (n_tri_int / n_tri_tot) if n_tri_tot > 0 else 0.0
        out_conf[qi] = np.log1p(len(common))       # log1p: keeps scale comparable to existing matchup_confidence

        pct = int(100 * (qi + 1) / n_q)
        if pct != last_pct and pct % 10 == 0:
            con.print(f"    {pct}% ({qi+1:,}/{n_q:,})"); last_pct = pct

    q['raw_intransitivity_measure'] = out_meas
    q['raw_matchup_confidence']     = out_conf
    q['n_common_opponents']         = out_nco
    q['n_triads_total']             = out_ntot
    q['n_triads_intransitive']      = out_nint
    # Restore original pred_df order via _qidx
    q = q.sort_values('_qidx').reset_index(drop=True).drop(columns='_qidx')
    return q[['Date', 'Aname', 'Bname', 'gender_source',
              'raw_intransitivity_measure', 'raw_matchup_confidence',
              'n_common_opponents', 'n_triads_total', 'n_triads_intransitive']]


def load_hist(use_val: bool, genders: list, granularity: str) -> pd.DataFrame:
    """Load and concat ATP/WTA history with gender tag. VAL uses _VAL slice."""
    cols = {'sets': REQ_HIST_COLS_SETS, 'games': REQ_HIST_COLS_GAMES}.get(granularity, REQ_HIST_COLS_MATCH)
    atp_path = ATP_VAL_HIST if use_val else ATP_HIST
    wta_path = WTA_VAL_HIST if use_val else WTA_HIST
    parts = []
    if 'men' in genders:
        atp = pd.read_csv(atp_path, usecols=cols, low_memory=False)
        atp['gender_source'] = 'men'; parts.append(atp)
    if 'women' in genders:
        wta = pd.read_csv(wta_path, usecols=cols, low_memory=False)
        wta['gender_source'] = 'women'; parts.append(wta)
    return pd.concat(parts, ignore_index=True)


def main():
    np.random.seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument('--val',  action='store_true', help='Build VAL only')
    ap.add_argument('--test', action='store_true', help='Build TEST only')
    ap.add_argument('--gender', choices=['men', 'women', 'both'], default='both')
    ap.add_argument('--min-h2h', type=int, default=1,
                    help='Min prior H2H matches required on each triad edge (default 1 = any)')
    ap.add_argument('--granularity', choices=['matches', 'sets', 'games'], default='matches',
                    help='Win-count granularity: matches (default), sets, or games (finer dominance-direction signal)')
    args = ap.parse_args()
    build_val  = args.val  or not args.test
    build_test = args.test or not args.val
    genders = ['men', 'women'] if args.gender == 'both' else [args.gender]
    suffix = f'_h{args.min_h2h}' if args.min_h2h > 1 else ''   # don't clobber default-threshold runs
    if args.granularity != 'matches': suffix += f'_{args.granularity}'   # parallel files, never overwrite match-level

    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    for label, preds_path, base_out, use_val, enabled in [
        ('TEST', TEST_PREDS, OUT_TEST, False, build_test),
        ('VAL',  VAL_PREDS,  OUT_VAL,  True,  build_val),
    ]:
        if not enabled: continue
        out_path = base_out.replace('.csv', f'{suffix}.csv')
        con.print(f"[bold cyan]Raw IA: {label}  (min_h2h={args.min_h2h}, granularity={args.granularity})[/bold cyan]")
        pred = pd.read_csv(preds_path, usecols=REQ_PRED_COLS, low_memory=False)
        pred = pred[pred['gender_source'].isin(genders)].reset_index(drop=True)
        con.print(f"  Loaded {len(pred):,} pred rows from {preds_path} (genders={genders})")

        # Process per-gender so we don't cross-contaminate opponent sets
        parts = []
        for g in genders:
            g_pred = pred[pred['gender_source'] == g].reset_index(drop=True)
            g_hist = load_hist(use_val, [g], args.granularity)
            con.print(f"  [bold]{g}[/bold]: hist={len(g_hist):,}  preds={len(g_pred):,}")
            parts.append(compute_raw_ia(g_pred, g_hist, min_h2h=args.min_h2h, granularity=args.granularity))
        out = pd.concat(parts, ignore_index=True)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
        con.print(f"[bold green]-> {out_path}[/bold green]  ({len(out):,} rows)")

        # Summary
        tbl = Table(title=f"Raw IA distribution: {label}")
        tbl.add_column("Gender", justify="right")
        tbl.add_column("N", justify="right")
        tbl.add_column("Mean IA", justify="right")
        tbl.add_column("% IA>0", justify="right")
        tbl.add_column("Mean #common opp", justify="right")
        tbl.add_column("Median #triads", justify="right")
        for g in genders:
            sl = out[out['gender_source'] == g]
            tbl.add_row(g, f"{len(sl):,}",
                        f"{sl['raw_intransitivity_measure'].mean():.4f}",
                        f"{(sl['raw_intransitivity_measure'] > 0).mean() * 100:.1f}%",
                        f"{sl['n_common_opponents'].mean():.1f}",
                        f"{int(sl['n_triads_total'].median())}")
        con.print(tbl)


if __name__ == '__main__':
    main()
