"""Regenerate every table, figure and reported number (see README). Works on Linux, macOS and Windows.

Runs the scripts in paper order. Before each, prints what it reproduces in the paper and which of its
printed tables are diagnostics only; then prints the script's own output and keeps a copy in outputs/logs/.

Usage: python run_eval.py          # from the included predictions (about 2 min)
       python run_eval.py --full   # first retrain the model and rebuild the triad counts, overwriting
                                   # data/predictions/; also retrains the ablation (Table 3) and runs the I* shrinkage (GPU recommended)
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IA = 'data/scripts/raw_intransitivity.py'
# (log name, command, what it produces / reproduces in the paper, its printed tables that are diagnostics only)
BUILD = [('build_preds', ['model/scripts/build_preds_w_ia.py'],
          'Model predictions, data/predictions/preds_w_ia{,_val}.csv: input to every result below', '')]
BUILD += [(f'raw_ia{n}', [IA, *a], f'Triad counts{d}, data/predictions/ia/raw_ia{{,_val}}{n}.csv: the T_h variable of Section 6', '')
          for n, a, d in [('', [], ' (h=1; also the source of the Section 6.1 gender rates)'),
                          *((f'_h{h}', ['--min-h2h', str(h)], f' (h={h})') for h in (2, 3, 4)),
                          *((f'_{g}', ['--granularity', g], f' ({g} basis)') for g in ('sets', 'games'))]]
ABL = 'eval/ablation.py'
# In paper order: (command, when: 'both' | 'default' | 'full', what it reproduces in the paper, diagnostics-only tables)
SCRIPTS = [
    ('eval/default/data_summary.py', 'both', 'Section 3: matches, players, date span and validation/test splits', ''),
    ('eval/intr/graph_construct_values.py', 'both', 'Figure 1 edge values and the 54% in its caption (the figure is drawn by hand)', ''),
    ('eval/default/tuning_table.py', 'both', 'Table 2; tuned values quoted in Sections 4.1-4.4 (q, learning rate, retraining)', ''),
    ('eval/default/plot_pareto.py', 'both', 'Figure 3 (Pareto front of the 250 tuning trials)', ''),
    ([ABL, '--table'], 'default', 'Table 3 and Section 4.5, from the included results (--full retrains them)', 'Part A stepwise table'),
    (ABL, 'full', 'Table 3 and Section 4.5, retrained (one row needs data not included: kept from the included results)',
     'Part A stepwise table'),
    ('eval/default/accuracy.py', 'both', 'Table 4; headline accuracy and Brier in the abstract and conclusion', ''),
    ('eval/default/cal_curve.py', 'both', 'Figure 4 (calibration curves)', ''),
    ('eval/intr/models_diff.py', 'both', 'Section 5.1 Murphy reliability and resolution; Table 5 (pooled confusion) and McNemar test',
     'men/women confusion, graph-feature and IA-slice tables'),
    ('eval/intr/example_figs.py', 'both', 'Figures 5 and 6; the career head-to-heads in Section 1', ''),
    ('eval/intr/triadviz.py', 'both', 'Figure 6 edge weights (the "Subgraph edges" table)', 'cycle counts table'),
    ('eval/intr/distviz.py', 'both', 'Section 6.1 triad rates (13.87% vs 12.37%) and I* gender means; Figure 7 and its caption',
     'by-surface table'),
    ('eval/intr/stacker_test.py', 'both', 'Section 6.2 headline (Global rows: B(W+G) 0.2113 vs 0.2130, b_2 0.498)',
     'IA=0 / IA>0 rows (refit per slice; Table 8 uses one fixed fit)'),
    ('eval/intr/conflict_cycle.py', 'both', 'Table 6 (S4 cell rows); Section 6.2 T=0 / T>0 p-values (S4); Section 6.4 (S2, S3, focal conflict)',
     'S1, triad-unmeasurable row, h2/h3/games/sets variants'),
    ('eval/intr/scalar_family.py', 'both', 'Section 6.2 scalar rivals (Partner-uniqueness table)', 'family stacks, stack DM tests, robustness'),
    ('eval/intr/pin_encompassing.py', 'both', 'Section 6.2 market encompassing (h=1 pooled rows)', 'h=2, h=3 and per-gender rows'),
    ('eval/intr/h2h_dist.py', 'both', 'Table 7 (pair shares also in outputs/h2h_dist.csv)', ''),
    ('eval/intr/threshold_sens.py', 'both', 'Table 8', ''),
    ('eval/intr/triad_granularity_compare.py', 'both', 'Section 6.3 cross-basis agreement (93-96%)', 'encompassing by basis (refit per slice)'),
    ('eval/intr/game_icc.py', 'both', 'Section 6.3 intra-match correlation, effective games and flip probabilities (all-matches rows)',
     'same-surface rows'),  # before record_stability.py, which reads its rho
    ('eval/intr/graph_stability.py', 'both', 'Section 6.3 leave-one-match-out stability (the "all" row)', 'per-meeting-count rows'),
    ('eval/intr/record_stability.py', 'both', 'Section 6.3 label retention (two-stage column) and game-redraw worlds',
     'match-only, >=80% and LOMO columns'),
    ('eval/default/mathsport_table.py', 'both', 'Table B.9 and the Appendix B text', ''),
    ('eval/intr/dominance_shrinkage_games.py', 'both', 'Appendix D, first paragraph (Overall rows)', 'per-surface rows'),
    ('eval/intr/dominance_shrinkage_assembled.py', 'both', 'Appendix D, second paragraph (alpha=1 row)', 'alpha=3, 5 rows'),
    ('eval/intr/ia_shrinkage_robustness.py', 'full', 'Appendix D, third paragraph (weight-fixed variant)', 'weight-shrunk variant'),
]
ENV = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}    # rich tables on Windows consoles/logs


def run(name, args, show=True):  # run one script with this interpreter; output to outputs/logs/<name>.log (+ terminal if show)
    log = ROOT / 'outputs' / 'logs' / f'{name}.log'
    print(f"   > python {' '.join(args)}", flush=True)
    t = time.time()
    with open(log, 'w', encoding='utf-8') as f, subprocess.Popen(
            [sys.executable, *args], cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8') as proc:
        for line in proc.stdout:
            f.write(line)
            if show: print(line, end='', flush=True)
    if proc.returncode:
        sys.exit(f'FAILED, see {log.relative_to(ROOT)}')
    print(f'   done in {time.time() - t:.0f}s -> {log.relative_to(ROOT)}')


if __name__ == '__main__':
    full = sys.argv[1:] == ['--full']
    if sys.argv[1:] not in ([], ['--full']):
        sys.exit(__doc__)
    sys.stdout.reconfigure(errors='replace')  # consoles that cannot show box-drawing characters
    (ROOT / 'outputs' / 'logs').mkdir(parents=True, exist_ok=True)
    steps = [(n, a, p, d, False) for n, a, p, d in (BUILD if full else [])]  # training progress: log only
    steps += [(Path(a[0]).stem, a, p, d, True) for c, when, p, d in SCRIPTS if when in ('both', 'full' if full else 'default')
              for a in [[c] if isinstance(c, str) else c]]
    for i, (name, args, paper, diag, show) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {'Builds' if not show else 'Paper'}: {paper}")
        if diag: print(f"   Diagnostics only (not in the paper): {diag}")
        run(name, args, show)
