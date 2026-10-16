import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
PREDS = "data/predictions/preds_w_ia.csv"


def plot_calibration_curve(df, prob_col="model_prob_non_isolated", outcome_col="Awin",
                           title="Calibration Plot", save_path=None):
    """Calibration plot with Tukey's recursive-halving bins."""
    df = df.dropna(subset=[prob_col, outcome_col])
    
    # Extract relevant data
    y_true = df[outcome_col]
    y_prob = df[prob_col]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    # Plot individual predictions (strip plot)
    # Adding jitter to y-axis for better visualization
    jitter = 0.01
    y_true_jitter = y_true + np.random.default_rng(SEED).uniform(-jitter, jitter, size=len(y_true))
    ax.plot(
        y_prob, y_true_jitter, "o", color="gray", alpha=0.3, markersize=2, label="_nolegend_"
    )
    
    # Binning data using Tukey's recursive halving method.
    # This creates finer bins at the extremes of the probability range.
    bins = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1.0 + 1e-8]
    # bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-8]
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    # Avoid division by zero for empty bins
    nonzero = bin_total > 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    # Plot calibration curve
    ax.scatter(prob_pred, prob_true, s=bin_total[nonzero], color="black", zorder=3, label="Model")
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "gray", linestyle="-", label="Perfect calibration", zorder=1)

    # Set plot details
    ax.set_xlabel("model probability", fontsize=14)
    ax.set_ylabel("empirical frequency", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 1.1])
    ax.grid(False)
    # ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved -> {save_path}")


if __name__ == "__main__":
    preds = pd.read_csv(PREDS)
    for gender, label in [('men', 'Men'), ('women', 'Women')]:
        plot_calibration_curve(preds[preds.gender_source == gender], title=label,
                               save_path=f"outputs/calibration_plot_{label}.png")

