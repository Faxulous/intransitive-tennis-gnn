#!/usr/bin/env python3
"""Stability of the evidence-weighted intransitivity I*(A_uv) under Bayesian shrinkage.

The logit transform in I(A) can blow up near dominance 0/1. We test whether the
descriptive I* statistics (Section 6.1) survive shrinking every dominance score toward neutral
by the same Beta rule as Appendix D (dominance_shrinkage_assembled.py):

    D_shrunk(u,v) = weighted_sum(u,v) / (cumulative_weight(u,v) + 2*alpha)

I*(A_uv) = I(A_uv) * sqrt(cumulative_weight[(u,v)]) is computed exactly as production does
(walk_forward_validation -> calculate_match_local_intransitivity_extended + get_match_confidence_values
-> add_scaled_ia's int_attempt2_sqrt). We RUN the real walk-forward (per gender, per split: VAL =
ridgeline of 6.1, TEST = encompassing binning) with the GNN train/predict stubbed to no-ops — I(A) and
confidence depend only on the dominance graph, not on any learned weights — so the PIM bookkeeping is
byte-identical to production and the unshrunk I* reproduces the published artefacts (validated per
surface against preds_w_ia*.csv before the shrunk side is trusted).

The shrunk I(A)/confidence are captured at the SAME call site by wrapping the PIM intransitivity method:
at each test window we additionally rebuild the interaction matrices from D_shrunk and recompute, on the
identical PIM state. Per query row we keep (surface, unshrunk I(A)/conf, shrunk I(A)/conf per alpha);
shrunk-vs-unshrunk stability stats are paired row-wise, so no match-id join is needed.

For each alpha in {1,3,5} and each shrunk variant we report, over matches with defined unshrunk I*>0:
  - Kendall tau + Spearman rho of I* (and of raw I(A)) vs unshrunk,
  - classification-flip counts/% (I(A) crossing 1.0; I*>0 vs =0),
  - mean / median / p95 / p99 of |Delta I*| and |Delta I(A)|.
Two weight conventions for the sqrt term:
  - WEIGHT-FIXED: sqrt(cw) unchanged, isolating the logit/Hodge effect.
  - WEIGHT-SHRUNK: sqrt(cw+2a), the fully consistent posterior.

Run: python eval/intr/ia_shrinkage_robustness.py
"""
import os, sys, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from scipy.stats import kendalltau, spearmanr
from scipy.sparse import dok_matrix
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.graph.gnn import GraphEngine
from model.graph.player_interaction import PlayerInteractionMatrix
from model.walk_forward_validation import walk_forward_validation
from model.graph.ia_compute import (interaction_to_advantage_matrix, calculate_intransitivity_measure,
                                     extract_submatrix)
from model.seed import set_seed

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
CFG = 'tunes/tune_degree_postab_best.json'          # final degree-only T170 config
ALPHAS = [1, 3, 5]                                   # primary alpha=1 (Appendix D), also 3,5
CLIP_EPS = 1e-6                                       # interaction_to_advantage_matrix default
K_HOPS, MIN_COMMON = 1, 1                             # as in walk_forward_validation
INIT_EPOCHS, RETRAIN_EPOCHS, RETRAIN_EVERY = 100, 30, 38  # schedule (GNN stubbed; kept for loop parity)
ZERO_COLS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]  # degree-only ablation (parity w/ build_preds_w_ia)
PUB = {'val': 'data/predictions/preds_w_ia_val.csv', 'test': 'data/predictions/preds_w_ia.csv'}
PAPER_VAL = {('men','Hard'):2.78,('women','Hard'):3.20,('men','Clay'):2.49,
             ('women','Clay'):2.33,('men','Grass'):1.69,('women','Grass'):2.10}
OUT_CSV = 'outputs/ia_shrinkage_robustness.csv'

con = Console(width=175)
_ORIG_CSG = PlayerInteractionMatrix.create_surface_graph
_ORIG_IA = PlayerInteractionMatrix.calculate_match_local_intransitivity_extended
_CAP = []  # flat list of per-query-row dicts, filled by the wrapped PIM method


def build_kw(p):  # graph-construction params (GNN-only params still passed; stub ignores them)
    surfaces = {"Hard":  {"Hard": 1.0, "Clay": p['HC'], "Grass": p['HG']},
                "Clay":  {"Hard": p['CH'], "Clay": 1.0, "Grass": p['CG']},
                "Grass": {"Hard": p['GH'], "Clay": p['GC'], "Grass": 1.0}}
    tq = {'GS': 1.0, 'Finals': p['beta_Finals'], '1000': p['beta_1000'], '500': p['beta_500']}
    return {'lr': p['lr'], 'K': p['K'], 'hidden': p['hidden'], 'layers': p['layers'],
            'activation': p['activation'], 'q': p['q'],
            'granularity': p['granularity'], 'lambda_param': p['lambda_param'],
            'max_edge_age': p['max_edge_age'], 'min_dom_pctile': p['min_dom_pctile'],
            'surfaces': surfaces, 'tournament_quality': tq, 'mlp_head': False, 'n_edge_features': 0}


def make_zero_cols_patch(cols=ZERO_COLS):  # class-level degree-only patch (matches build_preds_w_ia)
    def csg(self, surface, current_time=None, _o=_ORIG_CSG, _c=cols):
        _o(self, surface, current_time)
        for c in _c:
            self.surface_graphs[surface]['X_real'][:, c] = 0.0
    return csg


def shrink_matrices(pim, alpha):
    """Rebuild signed |posterior| interaction matrices from stored ws/cw with cw -> cw+2a.

    Mirrors _update_pair: post = clip(ws/(cw+2a), -0.5, 0.5); store |post| in the dominant direction.
    Only edges already present in the (pruned) interaction matrix are kept, so production's pruning is
    preserved — we shrink the magnitude, not the graph topology.
    """
    out = {}
    for s in pim.surfaces:
        im, ws, cw = pim.interaction_matrices[s], pim.weighted_sum[s], pim.cumulative_weight[s]
        sm = dok_matrix(im.shape, dtype=np.float32)
        seen = set()
        for (i, j), w in im.items():
            if w == 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            e = float(ws.get((a, b), 0.0)); c = float(cw.get((a, b), 1e-8))
            post = max(-0.5, min(0.5, e / (c + 2 * alpha)))
            if post > 0:
                sm[(a, b)] = post
            elif post < 0:
                sm[(b, a)] = -post
        out[s] = sm
    return out


def local_ia(matrices, surface, query_edges):
    """I(A) per query edge on an interaction-matrix dict (faithful core of the production routine).

    1-hop common-opponent neighbourhood, logit advantage matrix, I(A) via Hodge; 0.0 when <min_common
    common opponents (code's no-triad flag). On unshrunk matrices this reproduces production I(A).
    """
    if surface not in matrices:
        return np.zeros(len(query_edges))
    csr = matrices[surface].tocsr()

    def neigh(p):
        n = set(csr[p].nonzero()[1]) | set(csr[:, p].nonzero()[0])
        n.discard(p)
        return n

    res = []
    for a, b in query_edges:
        a, b = int(a), int(b)
        common = list(neigh(a) & neigh(b))
        if len(common) >= MIN_COMMON:
            sub = extract_submatrix(csr, [a, b] + common)
            if sub.size > 0 and sub.shape[0] >= 3:
                A = interaction_to_advantage_matrix(sub, clip_epsilon=CLIP_EPS)
                res.append(calculate_intransitivity_measure(A)['intransitivity_measure'])
            else:
                res.append(0.0)
        else:
            res.append(0.0)
    return np.array(res)


def conf_on(matrices, cw_dict, surface, query_edges):
    """matchup_confidence: cw[(a,b)] iff the edge survives in `matrices`; else 0 (mirrors production)."""
    im, cw = matrices[surface], cw_dict[surface]
    vals = []
    for pa, pb in query_edges:
        a, b = (int(pa), int(pb)) if pa < pb else (int(pb), int(pa))
        has = im.get((a, b), 0.0) != 0 or im.get((b, a), 0.0) != 0
        vals.append(float(cw.get((a, b), 0.0)) if has else 0.0)
    return np.array(vals)


def install_capture():
    """Wrap calculate_match_local_intransitivity_extended: append per-row unshrunk + shrunk I(A)/conf.

    Production calls this once per test window on pred_pim (graph+train+val + earlier test windows,
    pre-fold). We call the original (its return drives the production columns — guarantees unshrunk match),
    then on the SAME PIM state compute shrunk I(A) + confidence per alpha and append per query row.
    """
    win = [0]  # per-call counter == one test window; query_edges is expanded per set within a window

    def wrapped(self, surface, query_edges, k_hops=1, min_common=1):
        out = _ORIG_IA(self, surface, query_edges, k_hops=k_hops, min_common=min_common)
        qe = query_edges.cpu().numpy().astype(int)
        ia0 = np.array(out['intransitivity_measure'])
        cf0 = conf_on(self.interaction_matrices, self.cumulative_weight, surface, qe)
        shr = {a: (lambda sm: (local_ia(sm, surface, qe),
                               conf_on(sm, self.cumulative_weight, surface, qe)))(shrink_matrices(self, a))
               for a in ALPHAS}
        for k in range(len(qe)):
            # match key = (window-call, sorted player pair): collapses per-set expansion to one row/match,
            # mirroring walk_forward's drop_duplicates(match_id, keep='first').
            a_, b_ = int(qe[k][0]), int(qe[k][1])
            rec = {'win': win[0], 'pair': (min(a_, b_), max(a_, b_)), 'surface': surface,
                   'ia0': ia0[k], 'cf0': cf0[k]}
            for a in ALPHAS:
                rec[f'ia{a}'] = shr[a][0][k]; rec[f'cf{a}'] = shr[a][1][k]
            _CAP.append(rec)
        win[0] += 1
        return out
    PlayerInteractionMatrix.calculate_match_local_intransitivity_extended = wrapped


def stub_gnn(ge):
    """No-op the GNN: training does nothing; prediction returns uniform 0.5 (I*/conf are GNN-free)."""
    ge.train_multi_surface = lambda *a, **k: 0.0
    ge.train = lambda *a, **k: 0.0
    ge.predict_window = lambda query_edges, surface, best_of=None: torch.full(
        (query_edges.size(0), 2), 0.5, device=query_edges.device)


def run(gender, split, kw):
    """Run the real walk-forward (GNN stubbed, degree-only patch) and return captured per-row records."""
    global _CAP
    _CAP = []
    set_seed(SEED)
    prefix = 'atp' if gender == 'men' else 'wta'
    suffix = '_VAL' if split == 'val' else ''
    df = pd.read_csv(f'data/{prefix}_cleaned{suffix}.csv')
    ge = GraphEngine(**kw, gender=gender)
    stub_gnn(ge)
    try:
        PlayerInteractionMatrix.create_surface_graph = make_zero_cols_patch()
        install_capture()
        walk_forward_validation(ge, df, gender, seed=SEED, verbose=False,
                                initial_epochs=INIT_EPOCHS, subsequent_epochs=RETRAIN_EPOCHS,
                                retrain_every=RETRAIN_EVERY, compute_intransitivity=True)
    finally:
        PlayerInteractionMatrix.create_surface_graph = _ORIG_CSG
        PlayerInteractionMatrix.calculate_match_local_intransitivity_extended = _ORIG_IA
    rep = pd.DataFrame(_CAP).drop_duplicates(['win', 'pair'], keep='first').reset_index(drop=True)
    rep['gender_source'] = gender
    rep['split'] = split
    return rep


def istar(ia, cw):  # I* = I(A)*sqrt(cumulative_weight); 0 (undefined) where no surviving direct edge
    return ia * np.sqrt(np.clip(cw, 0, None))


def compare(u_istar, u_ia, s_istar, s_ia):
    """Stability stats over the mask of unshrunk I*>0."""
    base = u_istar > 0
    d_is = np.abs(s_istar[base] - u_istar[base]); d_ia = np.abs(s_ia[base] - u_ia[base])
    return dict(
        n=int(base.sum()),
        tau_istar=kendalltau(u_istar[base], s_istar[base])[0], rho_istar=spearmanr(u_istar[base], s_istar[base])[0],
        tau_ia=kendalltau(u_ia[base], s_ia[base])[0], rho_ia=spearmanr(u_ia[base], s_ia[base])[0],
        flip_thr=int(np.sum((u_ia[base] > 1.0) != (s_ia[base] > 1.0))),
        flip_pos=int(np.sum((u_istar > 0) != (s_istar > 0))),
        mean_d_istar=d_is.mean(), med_d_istar=np.median(d_is),
        p95_d_istar=np.quantile(d_is, 0.95), p99_d_istar=np.quantile(d_is, 0.99),
        mean_d_ia=d_ia.mean(), med_d_ia=np.median(d_ia),
        p95_d_ia=np.quantile(d_ia, 0.95), p99_d_ia=np.quantile(d_ia, 0.99))


def validate_unshrunk(rep, gender, split):
    """Per-surface I*>0 mean of captured unshrunk I* vs published int_attempt2_sqrt (replay fidelity)."""
    pub = pd.read_csv(PUB[split])
    pub = pub[pub.gender_source == gender]
    rep = rep.copy()
    rep['istar0'] = istar(rep.ia0.values, rep.cf0.values)
    rows, ok = [], True
    for s in ['Hard', 'Clay', 'Grass']:
        mine = rep.loc[(rep.surface == s) & (rep.istar0 > 0), 'istar0'].mean()
        pubm = pub.loc[(pub.Surface == s) & (pub.int_attempt2_sqrt > 0), 'int_attempt2_sqrt'].mean()
        good = abs(mine - pubm) < 0.02
        ok &= good
        rows.append((s, mine, pubm, PAPER_VAL.get((gender, s)) if split == 'val' else None,
                     int(((rep.surface == s) & (rep.istar0 > 0)).sum()), good))
    return rows, ok


def main():
    con.print("I*(A_uv) shrinkage robustness  —  final T170 config")
    p = json.load(open(CFG))['params']
    kw = build_kw(p)
    max_logit = np.log((1 - CLIP_EPS) / CLIP_EPS)
    con.print(f"clip_epsilon = {CLIP_EPS:g}  ->  probabilities capped at 1-eps;  max |logit| = {max_logit:.3f}\n")

    # ── SANITY GATE: women VAL first, validate captured unshrunk I* against published means ──
    con.print("[yellow]Sanity gate[/yellow]: women VAL replay vs published int_attempt2_sqrt")
    rep_wv = run('women', 'val', kw)
    rows, ok = validate_unshrunk(rep_wv, 'women', 'val')
    tbl = Table(title="Captured unshrunk I* vs published (women VAL, I*>0)")
    for c in ['surface', 'my mean I*', 'published', 'paper target', 'n', 'match']:
        tbl.add_column(c, justify='right')
    for s, mine, pubm, tgt, n, good in rows:
        tbl.add_row(s, f'{mine:.3f}', f'{pubm:.3f}', f'{tgt:.2f}' if tgt else '-', str(n), 'OK' if good else 'OFF')
    con.print(tbl)
    if not ok:
        con.print("[bold red]SANITY GATE FAILED[/bold red]: captured unshrunk I* disagrees with published. Stopping.")
        return
    con.print("[bold green]Sanity gate PASSED[/bold green]: replay reproduces published I*.\n")

    # ── FULL: both genders, both splits ──
    reps = {('women', 'val'): rep_wv}
    for split in ['val', 'test']:
        for gender in ['men', 'women']:
            if (gender, split) in reps:
                continue
            con.print(f"[cyan]Replay {gender} {split}[/cyan]")
            reps[(gender, split)] = run(gender, split, kw)

    all_rows = []
    for (gender, split), rep in reps.items():
        rows, ok = validate_unshrunk(rep, gender, split)
        flag = 'OK' if ok else 'OFF'
        worst = max(abs(m - pm) for _, m, pm, _, _, _ in rows)
        con.print(f"  unshrunk fidelity {gender} {split}: {flag}  (max per-surface |mean diff| = {worst:.4f})")
        ia0, cf0 = rep.ia0.values, rep.cf0.values
        istar0 = istar(ia0, cf0)
        for a in ALPHAS:
            ia_s, cf_s = rep[f'ia{a}'].values, rep[f'cf{a}'].values
            istar_fix = istar(ia_s, cf0)                         # WEIGHT-FIXED: sqrt(cw)
            cf_shr = np.where(cf0 > 0, cf0 + 2 * a, 0.0)
            istar_shr = istar(ia_s, cf_shr)                      # WEIGHT-SHRUNK: sqrt(cw+2a)
            for variant, ist in [('weight-fixed', istar_fix), ('weight-shrunk', istar_shr)]:
                st = compare(istar0, ia0, ist, ia_s)
                st.update(gender=gender, split=split, alpha=a, variant=variant)
                st['flip_thr_pct'] = 100 * st['flip_thr'] / max(st['n'], 1)
                st['flip_pos_pct'] = 100 * st['flip_pos'] / max(len(istar0), 1)
                all_rows.append(st)

    res = pd.DataFrame(all_rows)
    os.makedirs('outputs', exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    for variant in ['weight-fixed', 'weight-shrunk']:
        tbl = Table(title=f"I* stability under shrinkage [{variant}]  (matches with unshrunk I*>0)")
        for c in ['gender', 'split', 'a', 'n', 'tau I*', 'rho I*', 'tau I(A)', 'rho I(A)',
                  'I(A) flips', 'I*>0 flips', 'mean|dI*|', 'p95|dI*|', 'p99|dI*|', 'mean|dI(A)|', 'p99|dI(A)|']:
            tbl.add_column(c, justify='right')
        for r in res[res.variant == variant].sort_values(['split', 'gender', 'alpha']).itertuples():
            tbl.add_row(r.gender, r.split, str(r.alpha), str(r.n),
                        f'{r.tau_istar:.3f}', f'{r.rho_istar:.3f}', f'{r.tau_ia:.3f}', f'{r.rho_ia:.3f}',
                        f'{r.flip_thr} ({r.flip_thr_pct:.1f}%)', f'{r.flip_pos} ({r.flip_pos_pct:.1f}%)',
                        f'{r.mean_d_istar:.3f}', f'{r.p95_d_istar:.3f}', f'{r.p99_d_istar:.3f}',
                        f'{r.mean_d_ia:.3f}', f'{r.p99_d_ia:.3f}')
        con.print(tbl)

    con.print(f"\n  saved -> {OUT_CSV}")
    con.print("[dim]D_shrunk = weighted_sum/(cumulative_weight+2a); I* = I(A)*sqrt(weight). "
              "weight-fixed holds sqrt(cw) (isolates logit/Hodge); weight-shrunk uses sqrt(cw+2a).[/dim]")


if __name__ == '__main__':
    main()
