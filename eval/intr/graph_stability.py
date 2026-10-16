#!/usr/bin/env python3
"""Exact leave-one-match-out stability of the active test-entry dominance graph.

The production graph is rebuilt in its two stages (graph+train, then validation).
Every contributing physical match is then deleted once from each active edge and
the pair history is replayed with the production decay and expiry rules. Outcomes
are reported separately as retained, opposite, tied, or absent. This conditions on
matches admitted by the fitted dominance-percentile rule and does not retrain the
GNN or assess end-to-end forecast robustness.

Run: python eval/intr/graph_stability.py [--quick]
"""
import argparse, json, os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.graph.gnn import GraphEngine
from model.graph.tq_buckets import get_tier
from model.graph.windowstacks import create_meta_stacks, split_df_by_surface

con = Console(width=130)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CFG = 'tunes/tune_degree_postab_best.json'
OUT = 'outputs/graph_stability.csv'
SEC_YR = 365 * 24 * 3600
TOL = 1e-12


def build_kw(p):
    surfaces = {'Hard': {'Hard': 1.0, 'Clay': p['HC'], 'Grass': p['HG']},
                'Clay': {'Hard': p['CH'], 'Clay': 1.0, 'Grass': p['CG']},
                'Grass': {'Hard': p['GH'], 'Clay': p['GC'], 'Grass': 1.0}}
    tq = {'GS': 1.0, 'Finals': p['beta_Finals'], '1000': p['beta_1000'], '500': p['beta_500']}
    return {'lr': p['lr'], 'K': p['K'], 'hidden': p['hidden'], 'layers': p['layers'],
            'activation': p['activation'], 'q': p['q'],
            'granularity': p['granularity'], 'lambda_param': p['lambda_param'],
            'max_edge_age': p['max_edge_age'], 'min_dom_pctile': p['min_dom_pctile'],
            'surfaces': surfaces, 'tournament_quality': tq,
            'mlp_head': False, 'n_edge_features': 0}


def append_calls(pim, meta, stages, chains, seen):
    """Reconstruct production updates while retaining each physical match ID."""
    if pim.granularity != 'games':
        raise ValueError(f'Paper LOMO requires games granularity, got {pim.granularity!r}')
    for src, stack in meta.items():  # Same nesting/order as update_pim_from_stacks
        for stage in stages:
            windows = getattr(stack, stage)
            if windows is None:
                raise ValueError(f'{src}/{stage}: windows are None')
            for w in windows:
                matches = w.matches.cpu().numpy().astype(int)
                results = w.results.cpu().numpy()
                mids = w.match_ids.cpu().numpy().astype(int)
                dom = w.game_pct.cpu().numpy()
                if not (len(matches) == len(results) == len(mids) == len(dom)):
                    raise RuntimeError(f'{src}/{stage}: inconsistent window arrays')

                dev = np.abs(dom - 0.5)
                valid = dev[~np.isnan(dev)]
                threshold = np.percentile(valid, 100 * pim.min_dom_pctile) if len(valid) and pim.min_dom_pctile > 0 else 0.0
                tier = pim.tournament_quality[get_tier(w.tourney_level)]

                for k, (a, b) in enumerate(matches):
                    if np.isnan(dom[k]) or (threshold > 0 and dev[k] < threshold):
                        continue
                    winner, loser = (a, b) if results[k] == 1 else (b, a)
                    d = float(dom[k] - 0.5)
                    i, j, d = (winner, loser, d) if winner < loser else (loser, winner, -d)
                    mid, t = int(mids[k]), float(w.timestamp)
                    sig = (src, int(i), int(j), t, d)
                    if mid in seen and seen[mid] != sig:
                        raise RuntimeError(f'match_id {mid} has inconsistent identity')
                    seen[mid] = sig

                    chains.setdefault((int(i), int(j)), []).append((mid, src, t, 1.0, d))
                    for dst in pim.surfaces:
                        if dst == src:
                            continue
                        correlation = pim.surfaces[src][dst]
                        if correlation > 0:
                            chains[(int(i), int(j))].append((mid, dst, t, correlation * tier, d))


def active_snapshot(pim):
    """Signed active edge weights exactly as supplied to the GNN."""
    active = {}
    for surface in pim.surfaces:
        graph = pim.surface_graphs[surface]
        if graph['edge_index'] is None or graph['edge_weight'] is None:
            raise RuntimeError(f'{surface}: production graph was not initialised')
        idx = graph['edge_index'].cpu().numpy()
        weights = graph['edge_weight'].cpu().numpy()
        if idx.shape != (2, len(weights)):
            raise RuntimeError(f'{surface}: malformed edge tensors')
        active[surface] = {}
        for (u, v), weight in zip(idx.T, weights):
            u, v = int(u), int(v)
            pair = (min(u, v), max(u, v))
            if pair in active[surface]:
                raise RuntimeError(f'{surface}: duplicate directions for pair {pair}')
            active[surface][pair] = float(weight if u < v else -weight)
    return active


def replay(seq, lam, entry, max_age, prior, skip=None):
    """Replay one surface history, then apply the production edge-age rule."""
    cw, ws, last = prior, 0.0, None
    for mid, t, c, d in seq:
        if mid == skip:
            continue
        decay = 1.0 if last is None else np.exp(np.clip(-lam * max(0, t - last) / SEC_YR, -700, 0))
        cw, ws, last = decay * cw + c, decay * ws + c * d, t
    if last is None or (max_age > 0 and last < entry - max_age * SEC_YR):
        return np.nan
    return ws / cw


def classify(values, baseline):
    values = np.asarray(values, dtype=float)
    absent = np.isnan(values)
    tied = ~absent & np.isclose(values, 0.0, atol=TOL, rtol=0)
    directed = ~absent & ~tied
    retained = directed & (np.sign(values) == np.sign(baseline))
    opposite = directed & ~retained
    out = {'retained': int(retained.sum()), 'opposite': int(opposite.sum()),
           'tied': int(tied.sum()), 'absent': int(absent.sum())}
    if sum(out.values()) != len(values):
        raise RuntimeError('LOMO categories are not exhaustive')
    return out


def toy_checks():
    entry, age = 3 * SEC_YR, 1.0
    seq = [(1, 0.0, 1.0, 0.2), (2, float(entry), 1.0, 0.1)]
    if not np.isnan(replay(seq, 0.3, entry, age, 1e-8, skip=2)):
        raise AssertionError('Removing recent evidence must expose an expired edge')
    if not np.isnan(replay([seq[1]], 0.3, entry, age, 1e-8, skip=2)):
        raise AssertionError('Removing a singleton must remove its edge')
    expected = {'retained': 1, 'opposite': 1, 'tied': 1, 'absent': 1}
    if classify([0.2, -0.2, 0.0, np.nan], 0.1) != expected:
        raise AssertionError('LOMO outcome classification failed')


def collect(gender, kw, quick):
    prefix = 'atp' if gender == 'men' else 'wta'
    df = pd.read_csv(f'data/{prefix}_cleaned.csv')
    if quick:
        df = df.head(4000)
    ge = GraphEngine(**kw, gender=gender)
    meta = create_meta_stacks(split_df_by_surface(df, 0.5, 0.1, 0.1), gender)
    chains, seen = {}, {}

    append_calls(ge.pim, meta, ['graph', 'train'], chains, seen)
    ge.pim.update_pim_from_stacks(meta, ['graph', 'train'])
    append_calls(ge.pim, meta, ['val'], chains, seen)
    ge.pim.update_pim_from_stacks(meta, ['val'])

    val_windows = [w for stack in meta.values() for w in stack.val]
    if not val_windows:
        raise RuntimeError(f'{gender}: no validation windows')
    entry = max(float(w.timestamp) for w in val_windows)
    active = active_snapshot(ge.pim)
    prior = max(ge.pim.kappa, 1e-8)

    rebuilt = {surface: {} for surface in ge.pim.surfaces}
    for pair, calls in chains.items():
        for surface in ge.pim.surfaces:
            seq = [(m, t, c, d) for m, s, t, c, d in calls if s == surface]
            if not seq:
                continue
            value = replay(seq, ge.pim.lambda_param, entry, ge.pim.max_edge_age, prior)
            if not np.isnan(value) and not np.isclose(value, 0.0, atol=TOL, rtol=0):
                rebuilt[surface][pair] = value

    for surface in active:
        if set(active[surface]) != set(rebuilt[surface]):
            missing = set(active[surface]) - set(rebuilt[surface])
            extra = set(rebuilt[surface]) - set(active[surface])
            raise RuntimeError(f'{gender}/{surface}: active-key mismatch ({len(missing)} missing, {len(extra)} extra)')
        for pair, value in active[surface].items():
            if not np.isclose(value, rebuilt[surface][pair], rtol=1e-4, atol=2e-7):
                raise RuntimeError(f'{gender}/{surface}/{pair}: active-weight mismatch')
    con.print(f'  [green]{gender}[/green]: {sum(map(len, active.values())):,} active edges; exact replay validated')
    return ge.pim, chains, active, entry


def show(out):
    tbl = Table(title='Exact leave-one-match-out stability of active edges')
    for col in ['meetings', 'edges', 'deletions', 'retained', 'opposite', 'tied', 'absent', 'edges with any non-retention']:
        tbl.add_column(col, justify='right')
    groups = [('1', out['meetings'] == 1), ('2', out['meetings'] == 2),
              ('3-4', out['meetings'].between(3, 4)), ('5+', out['meetings'] >= 5),
              ('all', np.ones(len(out), dtype=bool))]
    for label, mask in groups:
        d = out[mask]
        if d.empty:
            continue
        n = int(d['meetings'].sum())
        rates = [d[f'lomo_{x}'].sum() / n for x in ['retained', 'opposite', 'tied', 'absent']]
        tbl.add_row(label, f'{len(d):,}', f'{n:,}', *(f'{x:.1%}' for x in rates),
                    f"{d['lomo_any_nonretention'].mean():.1%}")
    con.print(tbl)
    multi = out[out['meetings'] >= 2]
    counts = multi[['lomo_retained', 'lomo_opposite', 'lomo_tied', 'lomo_absent']].sum()
    directed = counts['lomo_retained'] + counts['lomo_opposite']
    vulnerable = multi[multi['lomo_opposite'] > 0]
    stable = multi[multi['lomo_opposite'] == 0]
    tot = out[['lomo_retained', 'lomo_opposite', 'lomo_tied', 'lomo_absent']].sum()
    n = int(out['meetings'].sum())
    con.print(f"  All deletions: {tot['lomo_tied'] + tot['lomo_absent']:,} ({(tot['lomo_tied'] + tot['lomo_absent']) / n:.1%}) leave no "
              f"directed edge, {tot['lomo_retained']:,} ({tot['lomo_retained'] / n:.1%}) retain the direction, "
              f"{tot['lomo_opposite']:,} ({tot['lomo_opposite'] / n:.1%}) reverse it")
    con.print(f"  Multi-meeting deletions retaining the original direction when an edge remains directed: "
              f"{counts['lomo_retained'] / directed:.1%}")
    con.print(f"  Mean |D-0.5|: {vulnerable['abs_dominance'].mean():.3f} for edges with an opposite-direction "
              f"deletion vs {stable['abs_dominance'].mean():.3f} otherwise")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Use the first 4,000 rows per tour')
    args = ap.parse_args()
    toy_checks()
    with open(CFG, encoding='utf-8') as f:
        p = json.load(f)['params']
    kw = build_kw(p)
    rows = []
    con.print('Rebuild and validate the production test-entry graph')
    for gender in ['men', 'women']:
        pim, chains, active, entry = collect(gender, kw, args.quick)
        prior = max(pim.kappa, 1e-8)
        for surface, graph in active.items():
            for pair, baseline in graph.items():
                calls = chains[pair]
                seq = [(m, t, c, d) for m, s, t, c, d in calls if s == surface]
                mids = list(dict.fromkeys(m for m, *_ in seq))
                if not mids:
                    raise RuntimeError(f'{gender}/{surface}/{pair}: active edge has no history')
                values = [replay(seq, pim.lambda_param, entry, pim.max_edge_age, prior, skip=m) for m in mids]
                counts = classify(values, baseline)
                n = len(mids)
                cutoff = entry - pim.max_edge_age * SEC_YR
                recent = len({m for m, t, *_ in seq if pim.max_edge_age <= 0 or t >= cutoff})
                if recent < 1:
                    raise RuntimeError(f'{gender}/{surface}/{pair}: active edge has no recent meeting')
                rows.append({'gender': gender, 'surface': surface, 'player_i': pair[0], 'player_j': pair[1],
                             'meetings': n, 'recent_meetings': recent, 'abs_dominance': abs(baseline),
                             **{f'lomo_{k}': v for k, v in counts.items()},
                             **{f'lomo_{k}_rate': v / n for k, v in counts.items()},
                             'lomo_any_nonretention': counts['opposite'] + counts['tied'] + counts['absent'] > 0})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError('LOMO produced no active edges')
    con.print('Results')
    con.print(f"  {len(out):,} active edges; {(out['meetings'] == 1).mean():.1%} have one meeting in total; "
              f"{(out['recent_meetings'] == 1).sum():,} ({(out['recent_meetings'] == 1).mean():.1%}) have one meeting within the activity window")
    show(out)
    path = OUT.replace('.csv', '_quick.csv') if args.quick else OUT
    os.makedirs('outputs', exist_ok=True)
    out.to_csv(path, index=False)
    con.print(f'\n  per-edge results -> {path}')


if __name__ == '__main__':
    main()
