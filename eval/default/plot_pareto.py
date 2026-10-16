import json, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D # Import Line2D for custom legend
import seaborn as sns
import numpy as np

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points. A point is Pareto efficient if no other point
    dominates it. A point A dominates B if A is strictly better on at least one
    objective and not worse on any other.
    """
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        # Check if any other point dominates point i
        is_dominated = np.any(np.all(costs[i] >= costs, axis=1) & np.any(costs[i] > costs, axis=1))
        if is_dominated:
            is_efficient[i] = False
            
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

def plot_pareto_front(csv_path='tunes/tune_degree_postab_trials.csv',  # final 250-trial re-tune (T170)
                      output_path='outputs/pareto_front.png',
                      best_json='tunes/tune_degree_postab_best.json',
                      chosen_trial_number=None):
    """
    Generates and saves a Pareto front plot from hyperparameter tuning trials.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # --- Data Preparation ---
    df = df[df['state'] == 'COMPLETE'].copy()
    if df.empty:
        print("No completed trials found in the data.")
        return
        
    df['men_brier'] = pd.to_numeric(df['values_0'])
    df['women_brier'] = pd.to_numeric(df['values_1'])
    df['neighborhood_size'] = df['params_K'] * df['params_layers']

    if chosen_trial_number is None:                            # Pull from best.json
        chosen_trial_number = json.load(open(best_json))['trial_number']

    # --- Pareto Front Calculation ---
    costs = df[['men_brier', 'women_brier']].values
    pareto_mask = is_pareto_efficient(costs)
    pareto_df = df[pareto_mask].sort_values('men_brier')

    # Identify the chosen trial by its number
    try:
        chosen_trial = df[df['number'] == chosen_trial_number].iloc[0]
    except IndexError:
        print(f"Error: Trial {chosen_trial_number} not found in the data.")
        pareto_df['brier_sum'] = pareto_df['men_brier'] + pareto_df['women_brier']
        chosen_trial = pareto_df.loc[pareto_df['brier_sum'].idxmin()]
        print(f"Highlighting trial {int(chosen_trial['number'])} instead.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(10, 8))

    # **FIX**: Create a custom dictionary to map neighborhood sizes to specific colors.
    unique_sizes = sorted(df['neighborhood_size'].unique())
    # Use a colorblind-friendly palette for the other sizes
    other_colors = sns.color_palette("colorblind", n_colors=len(unique_sizes))
    color_map = {size: color for size, color in zip(unique_sizes, other_colors)}
    
    # **Specifically set neighborhood size 2 to yellow/gold.**
    if 2 in color_map:
        color_map[2] = 'gold'

    # Scatter plot for all trials
    sns.scatterplot(
        data=df,
        x='men_brier',
        y='women_brier',
        hue='neighborhood_size',
        palette=color_map,       # Use the custom color dictionary
        ax=ax,
        s=50,
        alpha=0.6,
        legend='full'
    )

    # Plot Pareto front line
    ax.plot(pareto_df['men_brier'], pareto_df['women_brier'], 'k--', alpha=0.7, lw=2)

    # Re-plot Pareto points with bold outlines
    sns.scatterplot(
        data=pareto_df,
        x='men_brier',
        y='women_brier',
        hue='neighborhood_size',
        palette=color_map,       # Use the custom color dictionary
        ax=ax,
        s=80,
        edgecolor='black',
        linewidth=1.5,
        zorder=5,
        legend=False # Avoid duplicate legend entries
    )

    # Highlight the chosen trial using the same color map
    ax.scatter(
        chosen_trial['men_brier'],
        chosen_trial['women_brier'],
        marker='o',
        color=color_map.get(chosen_trial['neighborhood_size']), # Look up the color in our map
        s=150,
        edgecolor='black',
        linewidth=2.5,
        zorder=10,
    )

    # --- Final Touches ---
    ax.set_xlabel("Men's Brier Score", fontsize=14)
    ax.set_ylabel("Women's Brier Score", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0.210, 0.240)            # both axes capped at 0.24: crops 29 outlier trials
    ax.set_ylim(0.215, 0.240)            # women floor 0.215 (min trial 0.221) trims dead space
    
    # **FIX**: Create a robust, custom legend from our color map
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=size,
                              markerfacecolor=color, markersize=10)
                       for size, color in sorted(color_map.items())]
    legend = ax.legend(handles=legend_elements, title='Neighborhood Size', fontsize=11, ncol=2, frameon=True)
    plt.setp(legend.get_title(), fontsize='13')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved -> {output_path}  (chosen trial T{int(chosen_trial['number'])}: "
          f"men={chosen_trial['men_brier']:.5f}, women={chosen_trial['women_brier']:.5f})")

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    plot_pareto_front()