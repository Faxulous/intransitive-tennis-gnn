#!/usr/bin/env python3
"""Table 2 (search spaces and selected values) and the tuning results quoted in Section 4.

Search spaces are read from the trial.suggest_* calls in model/scripts/tune_degree.py (the
final re-tune), selected values from tunes/tune_degree_postab_best.json (trial T170), and
the trial count from tunes/tune_degree_postab_trials.csv. Nothing is retrained.

Run: python eval/default/tuning_table.py
"""
import ast, json, os
import pandas as pd
from rich.console import Console
from rich.table import Table

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TUNE   = 'model/scripts/tune_degree.py'
BUILD  = 'model/scripts/build_preds_w_ia.py'   # applies LR_MULT to the tuned lr; fixes RETRAIN_EVERY
BEST   = 'tunes/tune_degree_postab_best.json'
TRIALS = 'tunes/tune_degree_postab_trials.csv'
TABLE2 = [('Model architecture', [('K', 'K, Chebyshev filter order'), ('hidden', 'Hidden units'), ('layers', 'Number of layers'),
                                  ('activation', 'Use activation function'), ('q', 'Directional weighting q')]),
          ('Dominance signal', [('granularity', 'Granularity'), ('min_dom_pctile', 'Min. dominance percentile')]),
          ('Time decay', [('lambda_param', 'lambda'), ('max_edge_age', 'Max. edge age')]),
          ('Surface transferability (symmetric)', [('HC', 'alpha_h,c'), ('HG', 'alpha_h,g'), ('CG', 'alpha_c,g')])]


def spaces(path):  # {param: search space} from trial.suggest_*('name', ...) calls
    out = {}
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '').startswith('suggest_'):
            name, *args = [ast.literal_eval(a) for a in n.args]
            out[name] = ', '.join(map(str, args[0])) if n.func.attr == 'suggest_categorical' else \
                f"{args[0]}-{args[1]}" + (' (log)' if any(k.arg == 'log' for k in n.keywords) else '')
    return out


def const(path, name):  # module-level constant from a script, without importing it (also from A, B = 1, 2)
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign):
            names = [e.id for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
            if name in names:
                v = ast.literal_eval(n.value)
                return v[names.index(name)] if isinstance(n.targets[0], ast.Tuple) else v
    raise KeyError(f'{name} not set in {path}')


def fmt(v): return f'{v:.2f}' if isinstance(v, float) else str(v)


def main():
    sp, best = spaces(TUNE), json.load(open(BEST))
    p, n = best['params'], len(pd.read_csv(TRIALS))
    con = Console(width=120)
    tab = Table(title=f"Table 2: search spaces and selected values (trial T{best['trial_number']} of {n})")
    for c in ['Group', 'Parameter', 'Search space', 'Selected']:
        tab.add_column(c, justify='right')
    for g, rows in TABLE2:
        for k, (key, lab) in enumerate(rows):
            tab.add_row(g if k == 0 else '', lab, sp[key], fmt(p[key]), end_section=k == len(rows) - 1)
    con.print(tab)
    if p['HC'] != p['CH'] or p['HG'] != p['GH'] or p['CG'] != p['GC']: raise ValueError('surface transfer is not symmetric')
    tiers = {k: p[k] for k in p if k.startswith('beta_')}
    lr_mult = const(BUILD, 'LR_MULT')
    tab = Table(title="Section 4: other tuning results quoted in the text")
    for c in ['Quantity', 'Value', 'Search space', 'Where']:
        tab.add_column(c, justify='right')
    tab.add_row('Hyperparameters in Table 2', str(sum(len(r) for _, r in TABLE2)), '', 'Section 4.4')
    tab.add_row('Tuning trials', f'{n:,}', '', 'Section 4.4, Figure 3')
    tab.add_row('Directional weighting q', f"{p['q']:.3f}", sp['q'], 'Section 4.2')
    tab.add_row('Adam learning rate (tuned lr x LR_MULT)', f"{p['lr'] * lr_mult:.1e}", f"{sp['lr']} x {lr_mult}", 'Section 4.2')
    tab.add_row('Retrain every (snapshots)', str(const(BUILD, 'RETRAIN_EVERY')), 'fixed', 'Section 4.3')
    tab.add_row('Max. edge age (years)', f"{p['max_edge_age']:.2f}", sp['max_edge_age'], 'Section 4.1')
    tab.add_row('Tournament-tier weights beta', ', '.join(f'{v:g}' for v in tiers.values()), 'fixed (ablation)', 'Table 2 notes')
    tab.add_row('Validation Brier, men / women', f"{best['brier_men_val']:.4f} / {best['brier_women_val']:.4f}", '', 'Figure 3')
    con.print(tab)


if __name__ == '__main__':
    main()
