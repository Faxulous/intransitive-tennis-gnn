import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from rich.console import Console
from rich.table import Table

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

con, COL = Console(), 'int_attempt2_sqrt'  # COL = I(A)

# Load data with both calculated and zero values for complete analysis
df_all = pd.read_csv('data/predictions/preds_w_ia.csv')  # test-set I* distribution (descriptive)
df_all = df_all[df_all['int_attempt2_sqrt'].notna()]
df_all = df_all[df_all['Surface'].notna()]

# Create filtered dataset (calculated values only) for visualization
df = df_all[df_all['int_attempt2_sqrt'] > 0.0]

def triad_rates_by_gender():  # Section 6.1: share of intransitive common-opponent triads (h=1, test set)
    r = pd.read_csv('data/predictions/ia/raw_ia.csv').groupby('gender_source')[['n_triads_intransitive', 'n_triads_total']].sum()
    r['share'] = r['n_triads_intransitive'] / r['n_triads_total']
    tab = Table(title=f"Intransitive common-opponent triads, test set (women {r.share['women'] / r.share['men'] * 100 - 100:.1f}% more than men)")
    for c in ['Gender', 'Intransitive triads', 'All triads', 'Intransitive %']: tab.add_column(c, justify='right')
    for g, x in r.iterrows(): tab.add_row(g, f"{x.n_triads_intransitive:,.0f}", f"{x.n_triads_total:,.0f}", f"{x.share * 100:.2f}")
    con.print(tab)

def create_surface_summary_table():  # Section 6.1: I* by surface; statistics on I(A)>0 only
    """Summary of I(A) by surface (I(A)=0: too few common opponents to measure)."""
    tab = Table(title="Test-set I(A) by surface (Mean/Median/Std on I(A)>0)")
    for c in ['Surface', 'Total', 'I(A)=0', 'I(A)>0', 'Zero %', 'Mean', 'Median', 'Std']:
        tab.add_column(c, justify='right')
    for surface, a in [*df_all.groupby('Surface'), ('Total', df_all)]:
        v = a.loc[a[COL] > 0, COL]
        tab.add_row(surface, f"{len(a):,}", f"{(a[COL] == 0).sum():,}", f"{len(v):,}", f"{(a[COL] == 0).mean()*100:.1f}",
                    f"{v.mean():.3f}", f"{v.median():.3f}", f"{v.std():.3f}", end_section=surface == 'Hard')
    con.print(tab)
    tab = Table(title="Mean I(A)>0 by gender")
    for c in ['Gender', 'N', 'Mean']: tab.add_column(c, justify='right')
    for g, v in df.groupby('gender_source')[COL]: tab.add_row(g, f"{len(v):,}", f"{v.mean():.2f}")
    con.print(tab)

def ridgeline_plot_by_surface():
    """Plot single ridgeline plot of I(A) distribution by surface and gender."""
    intransitivity_col = COL
    plot_df = df.copy()  # df holds I(A)>0 only
    surfaces = ['Grass', 'Clay', 'Hard']
    genders = ['men', 'women']
    tab = Table(title=f"Figure 7 I(A)>0 by surface and gender (N={len(plot_df):,}, range "
                      f"[{plot_df[COL].min():.3f}, {plot_df[COL].max():.3f}], mean {plot_df[COL].mean():.3f})")
    for c in ['Surface', 'Gender', 'N', 'Mean']: tab.add_column(c, justify='right')
    for surface in surfaces:
        for gender in genders:
            subset = plot_df[(plot_df['Surface'] == surface) & (plot_df['gender'] == gender)]
            tab.add_row(surface if gender == 'men' else '', gender, f"{len(subset):,}", f"{subset[COL].mean():.2f}")
    con.print(tab)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    gender_colors = {'men': '#0072BD', 'women': '#D95319'}
    ridge_height = 0.8
    ridge_spacing = 1.6
    gender_offset = 0.15

    # Calculate global x-range
    all_values = []
    for surface in surfaces:
        for gender in genders:
            subset = plot_df[(plot_df['Surface'] == surface) & (plot_df['gender'] == gender)]
            if len(subset) > 0:
                all_values.extend(subset[intransitivity_col].values)

    x_min, x_max = min(all_values), max(all_values)
    x_padding = (x_max - x_min) * 0.15
    x_range = np.linspace(x_min - x_padding, x_max + x_padding, 400)

    y_positions = []
    surface_labels = []

    # Plot each surface with gender offsets
    for surface_idx, surface in enumerate(surfaces):
        base_y = surface_idx * ridge_spacing
        y_positions.append(base_y)
        surface_labels.append(surface)

        for gender in genders:
            subset = plot_df[(plot_df['Surface'] == surface) & (plot_df['gender'] == gender)]
            values = subset[intransitivity_col].values

            if len(values) > 15:
                y_offset = gender_offset if gender == 'women' else -gender_offset
                y_pos = base_y + y_offset

                kde = stats.gaussian_kde(values)
                density = kde(x_range)
                density = density / density.max() * ridge_height

                ax.fill_between(x_range, y_pos, density + y_pos,
                               color=gender_colors[gender], alpha=0.6,
                               label=f'{gender.capitalize()}' if surface_idx == 0 else "")
                ax.plot(x_range, density + y_pos,
                       color=gender_colors[gender], linewidth=2, alpha=0.8)

    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(surface_labels, fontsize=14)
    ax.set_xlabel('Evidence-weighted intransitivity $I^*$', fontsize=14)
    ax.set_ylabel('Surface', fontsize=14)

    ax.set_ylim(-ridge_spacing*0.4, (len(surfaces) - 1) * ridge_spacing + ridge_height + 0.6)

    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axisbelow(True)

    ax.tick_params(axis='y', left=False)
    ax.tick_params(axis='x', labelsize=12)

    # Create legend with sample sizes
    legend_labels = []
    for gender in genders:
        total_n = len(plot_df[plot_df['gender'] == gender])
        legend_labels.append(f'{gender.capitalize()} (n={total_n:,})')

    handles = [plt.Rectangle((0,0),1,1, color=gender_colors['men'], alpha=0.6),
               plt.Rectangle((0,0),1,1, color=gender_colors['women'], alpha=0.6)]
    ax.legend(handles, legend_labels, loc='upper right', fontsize=12, frameon=True, fancybox=False, edgecolor='black')

    # White background for publication
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig('outputs/int_ridgeline_by_surface.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('saved -> outputs/int_ridgeline_by_surface.png')

if __name__ == "__main__":
    triad_rates_by_gender()         # Section 6.1: 13.87% vs 12.37%
    create_surface_summary_table()  # Section 6.1: I* by surface and gender
    ridgeline_plot_by_surface()     # Figure 7
