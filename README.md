# Reproducibility package: Capturing Intransitive Dominance in Tennis Forecasting: A Graph Neural Network Approach

Code and data for the paper (International Journal of Forecasting, IJF-D-25-00979).

- **Assembled:** 16 October 2026
- **Authors and contact:** Lawrence Clegg (lawrence.clegg@bristol.ac.uk) and John Cartlidge (john.cartlidge@bristol.ac.uk), University of Bristol
- **Licence:** MIT for the code (`LICENSE`). Match data derived from [Tennis-Data](http://www.tennis-data.co.uk) are included with its permission.

## Setup

Clone the repository and work from its root:

```
git clone https://github.com/Faxulous/intransitive-tennis-gnn.git   # or: gh repo clone Faxulous/intransitive-tennis-gnn
cd intransitive-tennis-gnn
```

Requires **Python 3.10** (several pinned packages have no wheels for newer versions), on Linux, macOS or Windows (tested on Linux). Create an environment with conda ([Miniconda](https://docs.anaconda.com/miniconda/) is enough):

```
conda create --name tennis-gnn python=3.10
conda activate tennis-gnn
```

or with [uv](https://docs.astral.sh/uv/), which also downloads Python 3.10 if needed:

```
uv venv --python 3.10 --seed .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Then install torch for your hardware. With an NVIDIA GPU:

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

or CPU only:

```
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```
pip install -r requirements.txt
```

## Reproducing the paper

Once the environment is installed, the reproducibility check is one of two commands, run from the repository root.

**Quick check (2 min).** Regenerates every table, figure and number in the paper from the included predictions:

```
python run_eval.py
```

**Full check (16 min, GPU recommended).** First retrains the model and rebuilds the triad counts, overwriting `data/predictions/`. It then runs everything in the quick check, the ablation study (Table 3), and the shrinkage tests (Appendix D):

```
python run_eval.py --full
```

Figures and tables go to `outputs/`. Tables are also printed in the terminal.

Machine: AMD Ryzen 7 9700X (8 cores), 32 GB RAM, NVIDIA RTX 5070 (12 GB). Times were measured under Ubuntu 26.04.1 with torch 2.11.0+cu128.

The paper's numbers come from the included predictions. Retraining under a different system or library version shifts results ever so slightly (accuracy 0.657 to 0.656, encompassing z 8.91 to 8.86). Seeds cannot remove this, as PyTorch [does not guarantee identical results across releases, platforms, or CPU and GPU, even with identical seeds](https://docs.pytorch.org/docs/2.11/notes/randomness.html#:~:text=Completely%20reproducible%20results%20are%20not%20guaranteed). `git checkout data/predictions` restores the included files.

### Rebuilding the match files (optional, 7 min)

After downloading the Tennis-Data spreadsheets (see Data), run these, then `python run_eval.py --full`:

```
python data/scripts/loader.py              # spreadsheets -> data/{atp,wta}_cleaned{,_VAL}.csv
python data/scripts/compute_baselines.py   # adds Elo, WElo and Bradley-Terry probabilities
```

### Hyperparameter search (optional, 6 h)

```
python model/scripts/tune_degree.py   # 250 Optuna trials -> tunes/tune_degree_postab_{best.json,trials.csv}
```

## Data

**Raw (not included)**
- Tennis-Data spreadsheets, `data/raw_tennisdata/{atp,wta}/{2012..2025}.xls(x)`: free from [tennis-data.co.uk](http://www.tennis-data.co.uk/data.php), no registration, a few minutes. Save the 28 files with their original names (e.g. `atp/2012.xls`). The paper used a download from 10 June 2026.
- Player height, weight and handedness, from [tennisexplorer.com](https://www.tennisexplorer.com) player pages (Livesport s.r.o.), free, no registration, collected January 2026. Used only in the static-attribute row of Table 3, not in the final model. The site's terms forbid copying (clause 2.10) and scraping (clause 2.11) without the provider's written consent, so we cannot share them or give third parties access. To use them, add columns `height` (cm), `weight` (kg), `bmi` and `righthanded` (1/0) for the 670 men and 640 women in `data/player_data/{men,women}.csv` (gaps are filled with medians), then run `python eval/ablation.py`. Without them, that row prints as "–".

**Intermediary (included).** Built from the raw data by the scripts named. `python run_eval.py` starts from these, so every result can be verified without rebuilding them.
- `data/{atp,wta}_cleaned{,_VAL}.csv`: one row per match, from `loader.py` and `compute_baselines.py`. Derived from Tennis-Data, with permission.
- `data/predictions/preds_w_ia{,_val}.csv`: model, baseline and market probabilities for 12,052 test and 8,435 validation matches, from `model/scripts/build_preds_w_ia.py`.
- `data/predictions/ia/raw_ia*.csv`: per-match triad counts, from `data/scripts/raw_intransitivity.py`.
- `data/player_data/{men,women}.csv`: player names and graph node indices, from `loader.py`.
- `tunes/*`: selected hyperparameters and Optuna trials, from `model/scripts/tune_degree.py` (paper) and `tune_final.py` (first stage, used by the ablation).

**Other included data**
- `data/predictions/mathsport/*.csv`: predictions of the earlier model in Appendix B, whose code is not included.
- `data/examples/fig1_matches.csv`, `fig5_h2h.csv`: public match results behind Figures 1 and 5 (2024 Wimbledon and Paris Olympics; ATP head-to-heads).

## Tables and figures

| Item | Script | Output |
|---|---|---|
| Table 1 | none (notation) | |
| Table 2 | `eval/default/tuning_table.py` (search run by `model/scripts/tune_degree.py`) | `tunes/tune_degree_postab_best.json` |
| Table 3 | `eval/ablation.py --table` (retrained with --full; one row needs data not included) | `outputs/clean_ablation_val.csv` |
| Table 4 | `eval/default/accuracy.py` | `outputs/performance_summary.tex` |
| Table 5 | `eval/intr/models_diff.py` | `outputs/logs/models_diff.log` |
| Table 6 | `eval/intr/conflict_cycle.py` | `outputs/logs/conflict_cycle.log` |
| Table 7 | `eval/intr/h2h_dist.py` | `outputs/h2h_dist.csv` |
| Table 8 | `eval/intr/threshold_sens.py` | `outputs/threshold_sens.csv` |
| Table B.9 | `eval/default/mathsport_table.py` | `outputs/logs/mathsport_table.log` |
| Figure 1 | `eval/intr/graph_construct_values.py` (values for the hand-drawn schematic) | `outputs/logs/graph_construct_values.log` |
| Figure 2 | none (schematic) | |
| Figure 3 | `eval/default/plot_pareto.py` | `outputs/pareto_front.png` |
| Figure 4 | `eval/default/cal_curve.py` | `outputs/calibration_plot_*.png` |
| Figures 5, 6 | `eval/intr/example_figs.py` (Figure 6 weights from `eval/intr/triadviz.py`) | `outputs/*_int_example.jpg` |
| Figure 7 | `eval/intr/distviz.py` | `outputs/int_ridgeline_by_surface.png` |

## Numbers in the text

| Result | Script |
|---|---|
| Accuracy and Brier scores (abstract, §1, §5) | `eval/default/accuracy.py` |
| Encompassing gain over WElo (abstract, §6.2) | `eval/intr/stacker_test.py`; on subsets: `eval/intr/conflict_cycle.py` |
| Cyclic-triad rates by gender (abstract, §1, §6.1) | `eval/intr/distviz.py` (counts from `data/scripts/raw_intransitivity.py`) |
| Dataset counts and splits (§3) | `eval/default/data_summary.py` (match files from `data/scripts/loader.py`) |
| Murphy decomposition (§5.1) | `eval/intr/models_diff.py` |
| I\* gender gap (§6.1) | `eval/intr/distviz.py` |
| Share of matches between players who never met (§6.1, §6.3) | `eval/intr/h2h_dist.py` |
| Scalar-rating and market tests (§6.2) | `eval/intr/scalar_family.py`, `eval/intr/pin_encompassing.py` |
| Match/set/game agreement (§6.3) | `eval/intr/triad_granularity_compare.py` |
| Intra-match ICC and design effect (§6.3) | `eval/intr/game_icc.py` |
| Leave-one-match-out edge stability (§6.3) | `eval/intr/graph_stability.py` |
| Record-uncertainty stability (§6.3; run `game_icc.py` first) | `eval/intr/record_stability.py` |
| Conflict–cycle linkage (§6.4) | `eval/intr/conflict_cycle.py` |
| Shrinkage checks (Appendix D) | `eval/intr/dominance_shrinkage_games.py`, `dominance_shrinkage_assembled.py`, `ia_shrinkage_robustness.py` (--full) |

## Layout

```
data/       data files; data/scripts/ builds them
model/      graph construction (graph/), walk-forward training, tuning (scripts/)
eval/       one script per analysis (default/: performance; intr/: intransitivity)
tunes/      paper configuration (tune_degree_postab_best.json) and Optuna trials
outputs/    generated tables and figures
tests/      pytest suite (pytest 9.1.1)
run_eval.py runs every evaluation script
```
