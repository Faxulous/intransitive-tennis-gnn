import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
import os
from rich.console import Console
from rich.table import Table

def top2(vals, hi):  # (best, second) on 3-dp rounded values; ties share a mark, a tied best leaves no second
    r = sorted((round(v, 3) for v in vals), reverse=hi)
    return r[0], r[1] if r[1] != r[0] else None


def create_performance_table(df):
    """
    Create publication-ready performance table grouped by gender and surface.
    Returns both pandas DataFrame and LaTeX string.
    """
    # Ensure required columns exist
    required_cols = ['gender_source', 'Surface', 'Awin', 'model_prob', 'elo_all_prob',
                     'elo_weighted_prob', 'bt_prob', 'PSA_ip']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []

    # Process Men's data by surface
    for surface in ['Clay', 'Grass', 'Hard']:
        subset = df[(df['gender_source'] == 'men') & (df['Surface'] == surface)].copy()
        if len(subset) == 0:
            continue

        subset = subset.dropna(subset=['Awin', 'model_prob', 'elo_all_prob',
                                       'elo_weighted_prob', 'bt_prob', 'PSA_ip'])

        if len(subset) == 0:
            continue

        row = {
            'Gender': 'Men',
            'Surface': surface,
            'Count': len(subset),
            'Model_Acc': accuracy_score(subset['Awin'], subset['model_prob'] > 0.5),
            'Model_Brier': brier_score_loss(subset['Awin'], subset['model_prob']),
            'Elo_Acc': accuracy_score(subset['Awin'], subset['elo_all_prob'] > 0.5),
            'Elo_Brier': brier_score_loss(subset['Awin'], subset['elo_all_prob']),
            'WElo_Acc': accuracy_score(subset['Awin'], subset['elo_weighted_prob'] > 0.5),
            'WElo_Brier': brier_score_loss(subset['Awin'], subset['elo_weighted_prob']),
            'BT_Acc': accuracy_score(subset['Awin'], subset['bt_prob'] > 0.5),
            'BT_Brier': brier_score_loss(subset['Awin'], subset['bt_prob']),
            'PS_Acc': accuracy_score(subset['Awin'], subset['PSA_ip'] > 0.5),
            'PS_Brier': brier_score_loss(subset['Awin'], subset['PSA_ip']),
        }
        results.append(row)

    # Process Women's data by surface
    for surface in ['Clay', 'Grass', 'Hard']:
        subset = df[(df['gender_source'] == 'women') & (df['Surface'] == surface)].copy()
        if len(subset) == 0:
            continue

        subset = subset.dropna(subset=['Awin', 'model_prob', 'elo_all_prob',
                                       'elo_weighted_prob', 'bt_prob', 'PSA_ip'])

        if len(subset) == 0:
            continue

        row = {
            'Gender': 'Women',
            'Surface': surface,
            'Count': len(subset),
            'Model_Acc': accuracy_score(subset['Awin'], subset['model_prob'] > 0.5),
            'Model_Brier': brier_score_loss(subset['Awin'], subset['model_prob']),
            'Elo_Acc': accuracy_score(subset['Awin'], subset['elo_all_prob'] > 0.5),
            'Elo_Brier': brier_score_loss(subset['Awin'], subset['elo_all_prob']),
            'WElo_Acc': accuracy_score(subset['Awin'], subset['elo_weighted_prob'] > 0.5),
            'WElo_Brier': brier_score_loss(subset['Awin'], subset['elo_weighted_prob']),
            'BT_Acc': accuracy_score(subset['Awin'], subset['bt_prob'] > 0.5),
            'BT_Brier': brier_score_loss(subset['Awin'], subset['bt_prob']),
            'PS_Acc': accuracy_score(subset['Awin'], subset['PSA_ip'] > 0.5),
            'PS_Brier': brier_score_loss(subset['Awin'], subset['PSA_ip']),
        }
        results.append(row)

    # Process Combined (Both) data - All surfaces
    subset = df.copy()
    subset = subset.dropna(subset=['Awin', 'model_prob', 'elo_all_prob',
                                   'elo_weighted_prob', 'bt_prob', 'PSA_ip'])

    if len(subset) > 0:
        row = {
            'Gender': 'Both',
            'Surface': 'All',
            'Count': len(subset),
            'Model_Acc': accuracy_score(subset['Awin'], subset['model_prob'] > 0.5),
            'Model_Brier': brier_score_loss(subset['Awin'], subset['model_prob']),
            'Elo_Acc': accuracy_score(subset['Awin'], subset['elo_all_prob'] > 0.5),
            'Elo_Brier': brier_score_loss(subset['Awin'], subset['elo_all_prob']),
            'WElo_Acc': accuracy_score(subset['Awin'], subset['elo_weighted_prob'] > 0.5),
            'WElo_Brier': brier_score_loss(subset['Awin'], subset['elo_weighted_prob']),
            'BT_Acc': accuracy_score(subset['Awin'], subset['bt_prob'] > 0.5),
            'BT_Brier': brier_score_loss(subset['Awin'], subset['bt_prob']),
            'PS_Acc': accuracy_score(subset['Awin'], subset['PSA_ip'] > 0.5),
            'PS_Brier': brier_score_loss(subset['Awin'], subset['PSA_ip']),
        }
        results.append(row)

    perf_df = pd.DataFrame(results)

    # Generate LaTeX table with formatting
    latex_lines = []
    latex_lines.append(r"\begin{table}[tb]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \caption{Model Performance vs Baselines: Men's and Women's Results by Surface.}")
    latex_lines.append(r"    \label{tab:performance_summary}")
    latex_lines.append(r"    \resizebox{1.0\linewidth}{!}{%")
    latex_lines.append(r"    \begin{threeparttable}")
    latex_lines.append(r"    \begin{tabular}{l l r *{5}{r r}}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"            &        &       & \multicolumn{2}{c}{\bf{Model}} & \multicolumn{2}{c}{Elo} & \multicolumn{2}{c}{WElo} & \multicolumn{2}{c}{BT} & \multicolumn{2}{c}{PS} \\")
    latex_lines.append(r"    \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}")
    latex_lines.append(r"    Gender & Surface & {Count} & {Acc} & {Brier} & {Acc} & {Brier} & {Acc} & {Brier} & {Acc} & {Brier} & {Acc} & {Brier} \\")
    latex_lines.append(r"    \midrule")

    # Helper function to format value with bold/underline
    def format_value(val, top):  # top = (best, second) from top2
        val_str = f"{val:.3f}"
        if round(val, 3) == top[0]:
            return f"\\textbf{{{val_str}}}"
        elif round(val, 3) == top[1]:
            return f"\\underline{{{val_str}}}"
        else:
            return val_str

    # Process each row
    for idx, row in perf_df.iterrows():
        gender = row['Gender']
        surface = row['Surface']
        count = int(row['Count'])

        # Rank models only (exclude PS): accuracy higher is better, Brier lower is better
        acc = top2([row['Model_Acc'], row['Elo_Acc'], row['WElo_Acc'], row['BT_Acc']], hi=True)
        bri = top2([row['Model_Brier'], row['Elo_Brier'], row['WElo_Brier'], row['BT_Brier']], hi=False)

        model_acc_fmt, model_brier_fmt = format_value(row['Model_Acc'], acc), format_value(row['Model_Brier'], bri)
        elo_acc_fmt, elo_brier_fmt = format_value(row['Elo_Acc'], acc), format_value(row['Elo_Brier'], bri)
        welo_acc_fmt, welo_brier_fmt = format_value(row['WElo_Acc'], acc), format_value(row['WElo_Brier'], bri)
        bt_acc_fmt, bt_brier_fmt = format_value(row['BT_Acc'], acc), format_value(row['BT_Brier'], bri)

        ps_acc_fmt = f"{row['PS_Acc']:.3f}"
        ps_brier_fmt = f"{row['PS_Brier']:.3f}"

        # Build LaTeX row
        latex_row = f"    {gender:<6} & {surface:<6} & {count:<5} & {model_acc_fmt} & {model_brier_fmt}   & {elo_acc_fmt} & {elo_brier_fmt}   & {welo_acc_fmt} & {welo_brier_fmt}   & {bt_acc_fmt} & {bt_brier_fmt}   & {ps_acc_fmt} & {ps_brier_fmt} \\\\"
        latex_lines.append(latex_row)

        # Add midrule after Men's data and after Women's data
        if gender == 'Men' and surface == 'Hard':
            latex_lines.append(r"    \midrule")
        elif gender == 'Women' and surface == 'Hard':
            latex_lines.append(r"    \midrule")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular}")
    latex_lines.append(r"    \begin{tablenotes}")
    latex_lines.append(r"      \item[*] \hspace{-2.5pt} Notes. \textbf{Bold} indicates best performance and \underline{underlined} indicates second-best among the evaluated models (excluding PS benchmark) for each surface/gender combination, with values tied at three decimal places sharing a mark. Count refers to the number of matches in the out-of-sample test set. All metrics rounded to three decimal places. Elo represents the standard Elo rating system by \cite{FiveThirtyEightSerenaGOAT}. WElo represents the Elo extension by \cite{angelini2022weighted}. BT is the Bradley-Terry model of \cite{BradleyTerry}. PS represents Pinnacle Sports odds with \cite{shin1993} margin adjustment.")
    latex_lines.append(r"    \end{tablenotes}")
    latex_lines.append(r"    \end{threeparttable}")
    latex_lines.append(r"    }%end resizebox")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines) + "\n"

    return perf_df, latex_str


def comprehensive_analysis(df):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    lo, hi = df["Date"].min(), df["Date"].max()
    perf_df, latex_str = create_performance_table(df)
    tab = Table(title=f"Test-set performance by gender and surface, {lo:%Y-%m-%d} to {hi:%Y-%m-%d} "
                      f"({(hi - lo).days / 365.25:.2f} years)")
    for col in ["Gender", "Surface", "Count", "M-Acc", "M-Brier", "E-Acc", "E-Brier",
                "W-Acc", "W-Brier", "BT-Acc", "BT-Brier", "PS-Acc", "PS-Brier"]:
        tab.add_column(col, justify="right")
    for _, row in perf_df.iterrows():
        acc = top2([row[f"{m}_Acc"] for m in ("Model", "Elo", "WElo", "BT")], hi=True)
        bri = top2([row[f"{m}_Brier"] for m in ("Model", "Elo", "WElo", "BT")], hi=False)
        def fmt(val, top):  # best bold, second underlined (as in the paper; terminal only)
            r = round(val, 3)
            return f"[bold]{val:.3f}[/bold]" if r == top[0] else f"[u]{val:.3f}[/u]" if r == top[1] else f"{val:.3f}"
        tab.add_row(row["Gender"], row["Surface"], f"{int(row['Count']):,}",
                    *(fmt(row[f"{m}_{k}"], t) for m in ("Model", "Elo", "WElo", "BT") for k, t in (("Acc", acc), ("Brier", bri))),
                    f"{row['PS_Acc']:.3f}", f"{row['PS_Brier']:.3f}", end_section=row["Gender"] == "Women" and row["Surface"] == "Hard")
    Console(width=130).print(tab)
    os.makedirs("outputs", exist_ok=True)
    latex_path = "outputs/performance_summary.tex"
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"saved -> {latex_path}")
    return perf_df, latex_str



if __name__ == "__main__":
    df = pd.read_csv("data/predictions/preds_w_ia.csv").rename(columns={"model_prob_non_isolated": "model_prob"})
    comprehensive_analysis(df)
