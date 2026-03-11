import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Plotting and visualization utilities
# ------------------------------------------------------------------

Z = 1.96

COLOR_MAP = {
    "Black/African American": "#D55E00", # Dark orange
    "Hispanic/Latino": "#0072B2", # Dark blue
    "White": "#999999", # Gray
    "Asian": "#009E73" # Green
}

SENSITIVITY_COLOR_MAP = {
    "Baseline": "#E69F00",
    "Mixed": "#56B4E9",
    "Multiperson": "#CC79A7"
}

RACE_ORDER = ["Black/African American", "Hispanic/Latino", "White", "Asian"]



def _safe_visualization_setup(df, race_col):
    """
    Apply race ordering with minimal setup.
    """
    out = df.copy()
    if race_col in out.columns:
        out[race_col] = pd.Categorical(out[race_col], categories=RACE_ORDER, ordered=True)
    if "Year" in out.columns and race_col in out.columns:
        out = out.sort_values(["Year", race_col])
    return out



def _prep_df(df, primary_statute_level=None):
    """
    Optional filter by primary_statute_level.
    Applies race ordering via visualization_setup (or fallback).
    """
    d = df.copy()

    if primary_statute_level is not None:
        if "primary_statute_level" not in d.columns:
            raise ValueError("primary_statute_level filter requested, but df has no 'primary_statute_level' column.")
        d = d[d["primary_statute_level"] == primary_statute_level].copy()

    if "race_std" not in d.columns:
        raise ValueError("df must contain 'race_std' column.")

    d = _safe_visualization_setup(d, "race_std")
    return d



def visualize_policing(policing_analysis):
    df = policing_analysis.copy()
    race_col = "Perceived Race"
    
    from table_utils import visualization_setup
    df = visualization_setup(df, race_col=race_col)
    
    # Use centralized color map
    color_map = COLOR_MAP
    
    years = sorted(df["Year"].dropna().unique().tolist())
    
    def _create_figure(y_col, title, ylabel, show_errors=True, show_n=True):
        """Create a single publication-quality line plot with error bars and sample sizes"""
        fig, ax = plt.subplots(figsize=(10, 7))

        
        for race in color_map.keys():
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                
                # Apply horizontal offset
                x_vals = d["Year"]
                
                # Plot main line
                ax.plot(x_vals, d[y_col], 
                       marker="o", markersize=8,
                       linewidth=linewidth, linestyle=linestyle,
                       color=color_map[race], label=race)
                
                # Add error ribbons if available and requested
                if show_errors:
                    se_col = f"{y_col} SE"
                    if se_col in d.columns and d[se_col].notna().any():
                        # Calculate 95% confidence interval bounds
                        ci_lower = d[y_col] - d[se_col]*1.96
                        ci_upper = d[y_col] + d[se_col]*1.96
                        
                        # Create semi-transparent ribbon (lighter for hit rate)
                        ribbon_alpha = 0.12 if "Hit Rate" in y_col else 0.2
                        ax.fill_between(x_vals, ci_lower, ci_upper,
                                       color=color_map[race], 
                                       alpha=ribbon_alpha, linewidth=0)
                
                # Add sample size annotations if requested
                if show_n:
                    count_col = None
                    if "Search Rate" in y_col or "Searches per" in y_col:
                        count_col = "Search Count"
                    elif "Hit Rate" in y_col:
                        count_col = "Hit Count"
                    elif "Stops per" in y_col:
                        count_col = "Stop Count"
                    
                    if count_col and count_col in d.columns:
                        for idx, row in d.iterrows():
                            # Only show n counts for 2024
                            if row["Year"] == 2024:
                                n = int(row[count_col])
                                x_pos = row["Year"]
                                y_val = row[y_col]
                                
                                # Use smaller offset and add bbox for better visibility
                                ax.annotate(f'n={n:,}', 
                                          xy=(x_pos, y_val),
                                          xytext=(0, 12), textcoords='offset points',
                                          fontsize=8, ha='center', alpha=0.9,
                                          color=color_map[race],
                                          bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', 
                                                   edgecolor='none',
                                                   alpha=0.7))
        
        ax.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(title="Perceived Race", fontsize=10, title_fontsize=11, 
                 frameon=True, fancybox=True, shadow=False, 
                 loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Format y-axis
        if "Rate" in y_col:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        
        # Add extra space at top for labels
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + 0.15 * y_range)
        
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave room for legend below
        return fig
    
    # Create key figures
    figs = {}
    
    figs['stops'] = _create_figure(
        "Stops per 1,000",
        "Police Stop Rates by Perceived Race\n(per 1,000 residents)",
        "Stops per 1,000 Residents"
    )
    
    # Create combined search rate comparison figure
    def _create_search_comparison():
        """Create side-by-side comparison of search rate and searches per capita"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'wspace': 0.3})
        
        # Panel A: Searches per 1,000 (per capita)
        for race in color_map.keys():
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d["Searches per 1,000"].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                x_vals = d["Year"]
                
                # Plot main line
                ax1.plot(x_vals, d["Searches per 1,000"], 
                        marker="o", markersize=8,
                        linewidth=linewidth, linestyle=linestyle,
                        color=color_map[race], label=race)
                
                # Add error ribbons
                se_col = "Searches per 1,000 SE"
                if se_col in d.columns and d[se_col].notna().any():
                    ci_lower = d["Searches per 1,000"] - d[se_col]*1.96
                    ci_upper = d["Searches per 1,000"] + d[se_col]*1.96
                    ax1.fill_between(x_vals, ci_lower, ci_upper,
                                    color=color_map[race], alpha=0.15, linewidth=0)
                
                # Add sample size annotations
                if "Search Count" in d.columns:
                    for _, row in d.iterrows():
                        # Only show n counts for 2024
                        if row["Year"] == 2024:
                            n = int(row["Search Count"])
                            x_pos = row["Year"]
                            y_val = row["Searches per 1,000"]
                            ax1.annotate(f'n={n:,}', 
                                        xy=(x_pos, y_val),
                                        xytext=(0, 12), textcoords='offset points',
                                        fontsize=8, ha='center', alpha=0.9,
                                        color=color_map[race],
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                 facecolor='white', 
                                                 edgecolor='none',
                                                 alpha=0.7))
        
        ax1.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Searches per 1,000 Residents", fontsize=12, fontweight='bold')
        ax1.set_title("(A) Searches per Capita\n(per 1,000 residents)", 
                     fontsize=15, fontweight='bold', pad=15)
        ax1.set_xticks(years)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        
        # Add extra space at top for labels
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min
        ax1.set_ylim(y_min, y_max + 0.15 * y_range)
        
        # Panel B: Search Rate (conditional)
        for race in color_map.keys():
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d["Search Rate"].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                x_vals = d["Year"]
                
                # Plot main line
                ax2.plot(x_vals, d["Search Rate"], 
                        marker="o", markersize=8,
                        linewidth=linewidth, linestyle=linestyle,
                        color=color_map[race], label=race)
                
                # Add error ribbons
                se_col = "Search Rate SE"
                if se_col in d.columns and d[se_col].notna().any():
                    ci_lower = d["Search Rate"] - d[se_col]*1.96
                    ci_upper = d["Search Rate"] + d[se_col]*1.96
                    ax2.fill_between(x_vals, ci_lower, ci_upper,
                                    color=color_map[race], alpha=0.15, linewidth=0)
                
                # Add sample size annotations
                if "Search Count" in d.columns:
                    for _, row in d.iterrows():
                        # Only show n counts for 2024
                        if row["Year"] == 2024:
                            n = int(row["Search Count"])
                            x_pos = row["Year"]
                            y_val = row["Search Rate"]
                            ax2.annotate(f'n={n:,}', 
                                        xy=(x_pos, y_val),
                                        xytext=(0, 12), textcoords='offset points',
                                        fontsize=8, ha='center', alpha=0.9,
                                        color=color_map[race],
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                 facecolor='white', 
                                                 edgecolor='none',
                                                 alpha=0.7))
        
        ax2.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Search Rate", fontsize=12, fontweight='bold')
        ax2.set_title("(B) Conditional Search Rate\n(among those stopped)", 
                     fontsize=15, fontweight='bold', pad=15)
        # Legend will be added at figure level below
        ax2.set_xticks(years)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add extra space at top for labels
        y_min, y_max = ax2.get_ylim()
        y_range = y_max - y_min
        ax2.set_ylim(y_min, y_max + 0.15 * y_range)
        
        fig.suptitle('Police Search Patterns by Perceived Race', 
            fontsize=16, fontweight='bold', y=1.00, va='bottom')
        
        # Add centered legend below the plots
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, title="Perceived Race", 
          fontsize=10, title_fontsize=11,
          frameon=True, fancybox=True, shadow=False,
          loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
        
        # Use subplots_adjust instead of tight_layout to avoid warning
        fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12, wspace=0.3)
        return fig
    
    figs['search_comparison'] = _create_search_comparison()
    
    figs['hit_rate'] = _create_figure(
        "Hit Rate",
        "Contraband Hit Rate by Perceived Race\n(Outcome Test: Contraband Found Given Search)",
        "Hit Rate"
    )
    
    return figs



def plot_agency_black_white_hit_rates(agency_hit_df):
    """
    Create a publication-ready scatter plot comparing White and Black hit rates by agency.
    
    Points above the diagonal (higher Black hit rate) are colored green.
    Points below the diagonal (higher White hit rate) are colored red.
    Uses smart label positioning to avoid overlap.
    
    Parameters
    ----------
    agency_hit_df : DataFrame
        Output from rates_utils.summarize_agency_black_white_hit_rates()
        Expected columns: agency_name, White_Hit_Rate, Black_Hit_Rate, Avg_Search_Count
    
    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot with color-coded points and non-overlapping labels
    """
    d = agency_hit_df.copy()

    x = d["White_Hit_Rate"] * 100
    y = d["Black_Hit_Rate"] * 100

    # Scale point sizes based on average search count
    sizes = 100 + 3.5 * np.sqrt(d["Avg_Search_Count"])

    # Determine colors based on position relative to diagonal
    # Green if Black hit rate > White hit rate (above diagonal)
    # Red if White hit rate > Black hit rate (below diagonal)
    colors = []
    for xi, yi in zip(x, y):
        if yi > xi:
            colors.append("#009E73")  # Green (higher Black hit rate)
        else:
            colors.append("#D55E00")  # Red/Orange (higher White hit rate)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Create scatter plot with color-coding
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        alpha=0.7,
        c=colors,
        edgecolor="white",
        linewidth=1.5,
        zorder=3
    )

    # Calculate axis limits with some padding
    max_xy = np.nanmax(np.concatenate([x.to_numpy(), y.to_numpy()]))
    min_xy = np.nanmin(np.concatenate([x.to_numpy(), y.to_numpy()]))
    max_val = max(5, max_xy + 5)
    min_val = max(0, min_xy - 2)

    # Add 45-degree reference line (parity line)
    ax.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        linestyle="--", 
        linewidth=2.5, 
        color="gray", 
        alpha=0.5,
        label="Equal Hit Rates",
        zorder=2
    )

    # Try to use adjustText for smart label positioning, fall back to basic if not available
    try:
        from adjustText import adjust_text
        
        texts = []
        for _, row in d.iterrows():
            xi = row["White_Hit_Rate"] * 100
            yi = row["Black_Hit_Rate"] * 100
            
            text = ax.text(
                xi, yi,
                row["agency_name"],
                fontsize=9,
                alpha=0.9,
                zorder=4,
                ha='center',
                va='center'
            )
            texts.append(text)
        
        # Adjust text positions to avoid overlap
        adjust_text(
            texts,
            x=x.values,
            y=y.values,
            ax=ax,
            arrowprops=dict(
                arrowstyle='-',
                color='gray',
                alpha=0.5,
                lw=0.5
            ),
            expand_points=(1.5, 1.5),
            expand_text=(1.2, 1.2),
            force_points=(0.3, 0.3),
            force_text=(0.5, 0.5),
            lim=500
        )
        
    except ImportError:
        # Fallback: manual positioning with offset based on position
        from matplotlib.patheffects import withStroke
        
        for _, row in d.iterrows():
            xi = row["White_Hit_Rate"] * 100
            yi = row["Black_Hit_Rate"] * 100
            
            # Smart offset based on quadrant relative to center
            center_x = (min_val + max_val) / 2
            center_y = (min_val + max_val) / 2
            
            if xi > center_x and yi > center_y:
                xytext = (8, 8)
            elif xi > center_x and yi <= center_y:
                xytext = (8, -8)
            elif xi <= center_x and yi > center_y:
                xytext = (-8, 8)
            else:
                xytext = (-8, -8)
            
            text = ax.annotate(
                row["agency_name"],
                xy=(xi, yi),
                xytext=xytext,
                textcoords="offset points",
                fontsize=8,
                alpha=0.85,
                zorder=4,
                ha='center'
            )
            # Add white outline to text for better visibility
            text.set_path_effects([
                withStroke(linewidth=2, foreground="white", alpha=0.8)
            ])

    # Format axes
    ax.set_xlabel("White Hit Rate (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Black/African American Hit Rate (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Agency-Level Comparison: White vs. Black Hit Rates\n(Outcome Test for Searches)",
        fontsize=15,
        fontweight="bold",
        pad=20
    )
    
    # Improve grid
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # Ensure aspect ratio is equal (square plot)
    ax.set_aspect('equal', adjustable='box')
    
    # Create custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], linestyle='--', linewidth=2.5, color='gray', alpha=0.5, label='Equal Hit Rates'),
        Patch(facecolor='#009E73', alpha=0.7, edgecolor='white', linewidth=1.5, label='Black Hit Rate > White'),
        Patch(facecolor='#D55E00', alpha=0.7, edgecolor='white', linewidth=1.5, label='White Hit Rate > Black')
    ]
    
    ax.legend(
        handles=legend_elements,
        fontsize=10, 
        frameon=True, 
        fancybox=True, 
        shadow=False,
        loc='upper left'
    )

    plt.tight_layout()
    return fig



def create_combined_sensitivity_visualization(
    baseline_df,
    mixed_df,
    multiperson_df
):
    """
    Create a publication-ready two-panel sensitivity figure comparing:
    - Baseline
    - Mixed classification
    - Multiperson stops

    Panel A: Conditional Search Rate
    Panel B: Contraband Hit Rate
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    latest_year = baseline_df["Year"].max()

    baseline = baseline_df[baseline_df["Year"] == latest_year].copy()
    mixed = mixed_df[mixed_df["Year"] == latest_year].copy()
    multiperson = multiperson_df[multiperson_df["Year"] == latest_year].copy()

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian"
    ]

    baseline = baseline.set_index("Perceived Race").reindex(race_order).reset_index()
    mixed = mixed.set_index("Perceived Race").reindex(race_order).reset_index()
    multiperson = multiperson.set_index("Perceived Race").reindex(race_order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(
        f"Sensitivity Analysis: Baseline vs. Mixed Classification vs. Multiperson Stops ({latest_year})",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    x = np.arange(len(race_order))
    width = 0.24

    baseline_color = SENSITIVITY_COLOR_MAP["Baseline"]
    mixed_color = SENSITIVITY_COLOR_MAP["Mixed"]
    multiperson_color = SENSITIVITY_COLOR_MAP["Multiperson"]

    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    # --------------------
    # Panel A: Search Rate
    # --------------------
    base_search = baseline["Search Rate"] * 100
    mixed_search = mixed["Search Rate"] * 100
    multi_search = multiperson["Search Rate"] * 100

    base_search_se = baseline["Search Rate SE"] * 100 * 1.96
    mixed_search_se = mixed["Search Rate SE"] * 100 * 1.96
    multi_search_se = multiperson["Search Rate SE"] * 100 * 1.96

    x_base = x - width
    x_mixed = x
    x_multi = x + width

    axes[0].bar(
        x_base,
        base_search,
        width,
        label="Baseline",
        color=baseline_color,
        alpha=0.8,
        yerr=base_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(baseline_color)}
    )

    axes[0].bar(
        x_mixed,
        mixed_search,
        width,
        label="Mixed",
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(mixed_color)}
    )

    axes[0].bar(
        x_multi,
        multi_search,
        width,
        label="Multiperson",
        color=multiperson_color,
        alpha=0.8,
        yerr=multi_search_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(multiperson_color)}
    )

    max_y_search = 0
    for i, xi in enumerate(x):
        vals = [
            (x_base[i], base_search.iloc[i], base_search_se.iloc[i], baseline.iloc[i]["Search Count"]),
            (x_mixed[i], mixed_search.iloc[i], mixed_search_se.iloc[i], mixed.iloc[i]["Search Count"]),
            (x_multi[i], multi_search.iloc[i], multi_search_se.iloc[i], multiperson.iloc[i]["Search Count"]),
        ]

        for xpos, val, err, n in vals:
            if np.isfinite(val):
                axes[0].text(
                    xpos, val + err + 0.3,
                    f"{val:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=7, fontweight="bold"
                )
                max_y_search = max(max_y_search, val + err + 3)

    axes[0].set_title("(A) Conditional Search Rate", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_ylabel("Search Rate (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Perceived Race", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(race_order, rotation=0, ha="center", fontsize=10)
    axes[0].legend(fontsize=10, frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle=":", axis="y")
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0, max_y_search * 1.02)

    # -----------------
    # Panel B: Hit Rate
    # -----------------
    base_hit = baseline["Hit Rate"] * 100
    mixed_hit = mixed["Hit Rate"] * 100
    multi_hit = multiperson["Hit Rate"] * 100

    base_hit_se = baseline["Hit Rate SE"] * 100 * 1.96
    mixed_hit_se = mixed["Hit Rate SE"] * 100 * 1.96
    multi_hit_se = multiperson["Hit Rate SE"] * 100 * 1.96

    axes[1].bar(
        x_base,
        base_hit,
        width,
        label="Baseline",
        color=baseline_color,
        alpha=0.8,
        yerr=base_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(baseline_color)}
    )

    axes[1].bar(
        x_mixed,
        mixed_hit,
        width,
        label="Mixed",
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(mixed_color)}
    )

    axes[1].bar(
        x_multi,
        multi_hit,
        width,
        label="Multiperson",
        color=multiperson_color,
        alpha=0.8,
        yerr=multi_hit_se,
        capsize=4,
        error_kw={"linewidth": 1.5, "alpha": 0.4, "ecolor": darken_color(multiperson_color)}
    )

    max_y_hit = 0
    for i, xi in enumerate(x):
        vals = [
            (x_base[i], base_hit.iloc[i], base_hit_se.iloc[i], baseline.iloc[i]["Hit Count"]),
            (x_mixed[i], mixed_hit.iloc[i], mixed_hit_se.iloc[i], mixed.iloc[i]["Hit Count"]),
            (x_multi[i], multi_hit.iloc[i], multi_hit_se.iloc[i], multiperson.iloc[i]["Hit Count"]),
        ]

        for xpos, val, err, n in vals:
            if np.isfinite(val):
                axes[1].text(
                    xpos, val + err + 0.5,
                    f"{val:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=7, fontweight="bold"
                )
                max_y_hit = max(max_y_hit, val + err + 5)

    axes[1].set_title("(B) Contraband Hit Rate", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_ylabel("Hit Rate (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Perceived Race", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(race_order, rotation=0, ha="center", fontsize=10)
    axes[1].legend(fontsize=10, frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle=":", axis="y")
    axes[1].set_axisbelow(True)
    axes[1].set_ylim(0, max_y_hit * 1.02)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig

  
def plot_enhancement_rate_by_race(enhancement_by_primary):
    """
    Overall enhancement rates by race (aggregated across all statute levels and categories).
    """
    # Aggregate across all statute levels and categories
    summary = (
        enhancement_by_primary
        .groupby('race_std', as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Apply race ordering
    summary['race_std'] = pd.Categorical(summary['race_std'], categories=RACE_ORDER, ordered=True)
    summary = summary.sort_values('race_std')
    
    # Extract data
    races = summary['race_std'].astype(str).to_list()
    enh_rates = summary['Enhancement Rate'].to_numpy() * 100  # Convert to percentage
    se = summary['SE'].to_numpy() * 100
    ns = summary['N'].to_numpy()
    
    # Calculate 95% CI
    errors = se * Z
    
    # Helper function to darken a color
    def darken_color(color, factor=0.7):
        """Darken a hex color by multiplying RGB values by factor"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        darkened = tuple(c * factor for c in rgb)
        return darkened
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(races))
    colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]
    
    # Create bars without outline
    bars = ax.bar(  # noqa: F841
        x, enh_rates,
        color=colors,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    
    # Add error bars matching bar color but darker and more transparent
    for i, (rate, err, color) in enumerate(zip(enh_rates, errors, colors)):
        if np.isfinite(rate):
            ax.errorbar(
                i, rate,
                yerr=err,
                fmt='none',
                ecolor=darken_color(color),
                alpha=0.4,
                capsize=5,
                capthick=1.5,
                linewidth=1.5
            )
    
    # Set labels and title
    ax.set_ylabel('Enhancement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax.set_title(
        'Overall Enhancement Charge Rate by Race\n(with 95% Confidence Intervals)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=0, ha='center', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add sample sizes and rates as text above bars
    for i, (rate, err, n) in enumerate(zip(enh_rates, errors, ns)):
        if np.isfinite(rate):
            ax.text(
                i, rate + err + 1,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
    
    # Set y-axis limits to prevent label clipping
    ymax = np.nanmax(enh_rates + errors)
    if np.isfinite(ymax):
        ax.set_ylim(0, min(100, ymax * 1.2))
    else:
        ax.set_ylim(0, 100)
    
    fig.tight_layout()
    return fig



def plot_enhancement_rate_by_race_statute(enhancement_by_primary):
    """
    Enhancement rates by race, stratified by statute level (Felony vs Misdemeanor).
    Creates side-by-side panels with one for Misdemeanor and one for Felony, colored by race.
    """
    # Aggregate by race and statute level
    summary = (
        enhancement_by_primary
        .groupby(['race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Apply race ordering
    summary['race_std'] = pd.Categorical(summary['race_std'], categories=RACE_ORDER, ordered=True)
    summary = summary.sort_values(['race_std', 'primary_statute_level'])
    
    # Separate by statute level
    mis = summary[summary['primary_statute_level'] == 'Misdemeanor'].copy()
    fel = summary[summary['primary_statute_level'] == 'Felony'].copy()
    
    # Helper function to darken a color
    def darken_color(color, factor=0.7):
        """Darken a hex color by multiplying RGB values by factor"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        darkened = tuple(c * factor for c in rgb)
        return darkened
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.3})
    
    # ---- Panel A: Misdemeanor ----
    races_mis = mis['race_std'].astype(str).to_list()
    mis_rates = mis['Enhancement Rate'].to_numpy() * 100
    mis_se = mis['SE'].to_numpy() * 100
    mis_n = mis['N'].to_numpy()
    mis_errors = mis_se * Z
    
    x_mis = np.arange(len(races_mis))
    colors_mis = [COLOR_MAP.get(r, "#7f7f7f") for r in races_mis]
    
    # Create bars without outline
    ax1.bar(
        x_mis, mis_rates,
        color=colors_mis,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    
    # Add error bars matching bar color
    for i, (rate, err, color) in enumerate(zip(mis_rates, mis_errors, colors_mis)):
        if np.isfinite(rate):
            ax1.errorbar(
                i, rate,
                yerr=err,
                fmt='none',
                ecolor=darken_color(color),
                alpha=0.4,
                capsize=5,
                capthick=1.5,
                linewidth=1.5
            )
    
    # Add labels for Misdemeanor panel
    ax1.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Enhancement Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title('(A) Misdemeanor Charges', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_mis)
    ax1.set_xticklabels(races_mis, rotation=0, ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add sample sizes as text above bars
    max_y_mis = 0
    for i, (rate, err, n) in enumerate(zip(mis_rates, mis_errors, mis_n)):
        if np.isfinite(rate):
            y_pos = rate + err + 0.3
            ax1.text(
                i, y_pos,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y_mis = max(max_y_mis, y_pos + 2)
    
    ax1.set_ylim(0, min(100, max_y_mis))
    
    # ---- Panel B: Felony ----
    races_fel = fel['race_std'].astype(str).to_list()
    fel_rates = fel['Enhancement Rate'].to_numpy() * 100
    fel_se = fel['SE'].to_numpy() * 100
    fel_n = fel['N'].to_numpy()
    fel_errors = fel_se * Z
    
    x_fel = np.arange(len(races_fel))
    colors_fel = [COLOR_MAP.get(r, "#7f7f7f") for r in races_fel]
    
    # Create bars without outline
    ax2.bar(
        x_fel, fel_rates,
        color=colors_fel,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    
    # Add error bars matching bar color
    for i, (rate, err, color) in enumerate(zip(fel_rates, fel_errors, colors_fel)):
        if np.isfinite(rate):
            ax2.errorbar(
                i, rate,
                yerr=err,
                fmt='none',
                ecolor=darken_color(color),
                alpha=0.4,
                capsize=5,
                capthick=1.5,
                linewidth=1.5
            )
    
    # Add labels for Felony panel
    ax2.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Felony Charges', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_fel)
    ax2.set_xticklabels(races_fel, rotation=0, ha='center', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Add sample sizes as text above bars
    max_y_fel = 0
    for i, (rate, err, n) in enumerate(zip(fel_rates, fel_errors, fel_n)):
        if np.isfinite(rate):
            y_pos = rate + err + 0.3
            ax2.text(
                i, y_pos,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y_fel = max(max_y_fel, y_pos + 2)
    
    ax2.set_ylim(0, min(100, max_y_fel))
    
    # Add overall title
    fig.suptitle(
        'Enhancement Charge Rate by Race and Statute Level\n(with 95% Confidence Intervals)',
        fontsize=15,
        fontweight='bold',
        y=1.03
    )

    fig.text(
    0.505, 0.52, 'Enhancement Rate (%)',
    ha='center',
    va='center',
    rotation='vertical',
    fontsize=12,
    fontweight='bold'
    )
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.10, wspace=0.3)
    return fig



def plot_enhancement_rate_by_race_statute_category(enhancement_by_primary, top_categories=None):
    """
    For each charge category, create a 2-panel figure:
      (A) Misdemeanor charges (left) and (B) Felony charges (right),
    with bars colored by race (COLOR_MAP) and 95% CI error bars.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = enhancement_by_primary.copy()

    # Categories to plot
    if top_categories is not None:
        categories = list(top_categories)
        df = df[df["primary_charge_category"].isin(categories)].copy()
    else:
        categories = sorted(df["primary_charge_category"].dropna().unique().tolist())

    # Helper: darken a color for error bars
    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    figs = {}

    for category in categories:
        cat = df[df["primary_charge_category"] == category].copy()

        # Aggregate by race and statute
        summary = (
            cat.groupby(["race_std", "primary_statute_level"], as_index=False)
               .agg({"Enhanced": "sum", "N": "sum"})
        )

        # Guard: if empty, skip
        if summary.empty:
            continue

        summary["Enhancement Rate"] = summary["Enhanced"] / summary["N"]
        summary["SE"] = np.sqrt(summary["Enhancement Rate"] * (1 - summary["Enhancement Rate"]) / summary["N"])

        # Apply race ordering
        summary["race_std"] = pd.Categorical(summary["race_std"], categories=RACE_ORDER, ordered=True)
        summary = summary.sort_values(["race_std", "primary_statute_level"])

        mis = summary[summary["primary_statute_level"] == "Misdemeanor"].copy()
        fel = summary[summary["primary_statute_level"] == "Felony"].copy()

        # Determine which races exist in either panel, preserving RACE_ORDER
        present_races = set(mis["race_std"].dropna().astype(str)).union(
            set(fel["race_std"].dropna().astype(str))
        )
        races = [r for r in RACE_ORDER if r in present_races]

        # Build aligned series per race (so both panels share the same x order)
        def aligned_arrays(subdf):
            d = {str(r): row for _, row in subdf.iterrows() for r in [row["race_std"]]}
            rates = np.array([100 * (d[r]["Enhancement Rate"] if r in d else np.nan) for r in races], dtype=float)
            ses   = np.array([100 * (d[r]["SE"] if r in d else np.nan) for r in races], dtype=float)
            ns    = np.array([d[r]["N"] if r in d else 0 for r in races], dtype=float)
            errs  = ses * Z
            return rates, errs, ns

        mis_rates, mis_errs, mis_n = aligned_arrays(mis)
        fel_rates, fel_errs, fel_n = aligned_arrays(fel)

        x = np.arange(len(races))
        colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]

        # Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"wspace": 0.3})

        # ---- Panel A: Misdemeanor ----
        ax1.bar(x, np.nan_to_num(mis_rates, nan=0.0), color=colors, alpha=0.85, edgecolor="none", linewidth=0)

        for i, (rate, err, col, n) in enumerate(zip(mis_rates, mis_errs, colors, mis_n)):
            if np.isfinite(rate) and n > 0:
                ax1.errorbar(i, rate, yerr=err, fmt="none", ecolor=darken_color(col),
                             alpha=0.4, capsize=5, capthick=1.5, linewidth=1.5)

        ax1.set_title("(A) Misdemeanor Charges", fontsize=13, fontweight="bold", pad=15)
        ax1.set_xlabel("Canonical Race", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Enhancement Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(races, rotation=0, ha="center", fontsize=10)
        ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax1.set_axisbelow(True)

        max_y_mis = 0
        for i, (rate, err, n) in enumerate(zip(mis_rates, mis_errs, mis_n)):
            if np.isfinite(rate) and n > 0:
                y_pos = rate + err + 0.3
                ax1.text(i, y_pos, f"{rate:.1f}%\n(n={int(n):,})",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")
                max_y_mis = max(max_y_mis, y_pos + 2)
        # add a little headroom so text doesn't touch the top spine
        pad = 2  # percentage points; tweak to taste (e.g., 1.5–3)
        ymax_mis = max_y_mis if max_y_mis > 0 else 100
        ax1.set_ylim(0, min(100, ymax_mis + pad))

        # ---- Panel B: Felony ----
        ax2.bar(x, np.nan_to_num(fel_rates, nan=0.0), color=colors, alpha=0.85, edgecolor="none", linewidth=0)

        for i, (rate, err, col, n) in enumerate(zip(fel_rates, fel_errs, colors, fel_n)):
            if np.isfinite(rate) and n > 0:
                ax2.errorbar(i, rate, yerr=err, fmt="none", ecolor=darken_color(col),
                             alpha=0.4, capsize=5, capthick=1.5, linewidth=1.5)

        ax2.set_title("(B) Felony Charges", fontsize=13, fontweight="bold", pad=15)
        ax2.set_xlabel("Canonical Race", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(races, rotation=0, ha="center", fontsize=10)
        ax2.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax2.set_axisbelow(True)

        max_y_fel = 0
        for i, (rate, err, n) in enumerate(zip(fel_rates, fel_errs, fel_n)):
            if np.isfinite(rate) and n > 0:
                y_pos = rate + err + 0.3
                ax2.text(i, y_pos, f"{rate:.1f}%\n(n={int(n):,})",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")
                max_y_fel = max(max_y_fel, y_pos + 2)
            ymax_fel = max_y_fel if max_y_fel > 0 else 100
            ax2.set_ylim(0, min(100, ymax_fel + pad))
        # Suptitle
        fig.suptitle(
            f"{category} Enhancement Charge Rate by Race and Statute Level\n(with 95% Confidence Intervals)",
            fontsize=15,
            fontweight="bold",
            y=1.03
        )

        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.17, wspace=0.3)

        figs[category] = fig

    return figs



def plot_wobbler_overall_felony_rate(df):
    """
    Overall wobbler felony filing rate by race (95% CI).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    Z_ = globals().get("Z", 1.96)

    # Build table (same logic as your table function)
    wobblers = df[df["is_wobbler"]].copy()
    summary = (
        wobblers.groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Felony", "Misdemeanor"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["Total"] = summary["Felony"] + summary["Misdemeanor"]
    summary["Felony Rate"] = np.where(summary["Total"] > 0, summary["Felony"] / summary["Total"], np.nan)
    summary["Felony Rate SE"] = np.where(
        summary["Total"] > 0,
        np.sqrt(summary["Felony Rate"] * (1 - summary["Felony Rate"]) / summary["Total"]),
        np.nan,
    )

    # Enforce race order and drop races not present (optional)
    summary = summary.reindex(RACE_ORDER)

    races = summary.index.astype(str).tolist()
    rates = (summary["Felony Rate"].to_numpy(dtype=float) * 100)
    ses   = (summary["Felony Rate SE"].to_numpy(dtype=float) * 100)
    ns    = summary["Total"].to_numpy(dtype=float)
    errs  = ses * Z_

    def darken(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    colors = [globals().get("COLOR_MAP", {}).get(r, "#7f7f7f") for r in races]
    x = np.arange(len(races))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, np.nan_to_num(rates, nan=0.0), color=colors, alpha=0.85, edgecolor="none", linewidth=0)

    for i, (rate, err, col, n) in enumerate(zip(rates, errs, colors, ns)):
        if np.isfinite(rate) and n > 0:
            ax.errorbar(i, rate, yerr=err, fmt="none", ecolor=darken(col),
                        alpha=0.4, capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_title("Wobbler Charges: Felony Filing Rate by Race\n(with 95% Confidence Intervals)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Canonical Race", fontsize=12, fontweight="bold")
    ax.set_ylabel("Wobbler Charged as Felony (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(races, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    # bar labels (rate + n), with a little headroom
    max_y = 0
    for i, (rate, err, n) in enumerate(zip(rates, errs, ns)):
        if np.isfinite(rate) and n > 0:
            y_pos = rate + err + 0.7
            ax.text(i, y_pos, f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            max_y = max(max_y, y_pos + 2)

    pad = 2  # extra space above the highest label (percentage points)

    ymax = max_y if max_y > 0 else 100
    ax.set_ylim(0, min(100, ymax + pad))
    fig.tight_layout()
    return fig



def plot_wobbler_cleveland_dot(df, top_categories=None, sort_by="rate_overall"):
    """
    Cleveland dot plot for wobbler felony filing rate by charge category and race.

    If top_categories is provided, only those categories are plotted.
    Otherwise, all available categories are used.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    Z_ = globals().get("Z", 1.96)
    races = list(globals().get("RACE_ORDER", ["Black/African American", "Hispanic/Latino", "White", "Asian"]))
    color_map = globals().get("COLOR_MAP", {})

    # Filter to wobblers
    wobblers = df[df["is_wobbler"]].copy()

    # Aggregate counts by category x race x statute
    g = (
        wobblers.groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure statute columns exist
    for col in ["Felony", "Misdemeanor"]:
        if col not in g.columns:
            g[col] = 0

    g["Total"] = g["Felony"] + g["Misdemeanor"]
    g["Felony Rate"] = np.where(g["Total"] > 0, g["Felony"] / g["Total"], np.nan)
    g["SE"] = np.where(
        g["Total"] > 0,
        np.sqrt(g["Felony Rate"] * (1 - g["Felony Rate"]) / g["Total"]),
        np.nan,
    )
    g["lo"] = (g["Felony Rate"] - Z_ * g["SE"]) * 100
    g["hi"] = (g["Felony Rate"] + Z_ * g["SE"]) * 100
    g["x"] = g["Felony Rate"] * 100

    # Keep only races of interest
    g = g[g["race_std"].isin(races)].copy()
    g["race_std"] = pd.Categorical(g["race_std"], categories=races, ordered=True)

    # Use supplied category list if provided
    if top_categories is not None:
        categories = list(top_categories)
        existing = set(g["charge_category"].unique())
        categories = [c for c in categories if c in existing]
        g = g[g["charge_category"].isin(categories)].copy()
    else:
        # Automatic ordering based on all available categories
        cat_totals = g.groupby("charge_category")["Total"].sum().sort_values(ascending=False)

        overall_rate = g.groupby("charge_category")[["Felony", "Misdemeanor"]].sum()
        overall_rate["Total"] = overall_rate["Felony"] + overall_rate["Misdemeanor"]
        overall_rate["Rate"] = np.where(
            overall_rate["Total"] > 0,
            overall_rate["Felony"] / overall_rate["Total"],
            np.nan
        )

        if sort_by == "total":
            categories = cat_totals.index.tolist()
        elif sort_by == "rate_overall":
            categories = overall_rate["Rate"].sort_values(ascending=False).index.tolist()
        elif sort_by == "rate_black":
            blk = g[g["race_std"] == races[0]].set_index("charge_category")["Felony Rate"]
            categories = blk.sort_values(ascending=False).index.tolist()
        else:
            categories = sorted(cat_totals.index.tolist())

    # Final filter/order
    g = g[g["charge_category"].isin(categories)].copy()
    categories = list(categories)

    y_base = np.arange(len(categories))
    offsets = np.linspace(-0.32, 0.32, num=len(races))

    fig_h = max(6, 0.35 * len(categories) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    for i, y0 in enumerate(y_base):
        if i % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", alpha=1.0, zorder=0)

    for r_i, race in enumerate(races):
        sub = g[g["race_std"] == race].copy()
        sub = sub.set_index("charge_category").reindex(categories)
        y = y_base + offsets[r_i]

        x = sub["x"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        n = sub["Total"].to_numpy(dtype=float)

        for yi, xi, l, h, nn in zip(y, x, lo, hi, n):
            if np.isfinite(xi) and nn > 0 and np.isfinite(l) and np.isfinite(h):
                ax.hlines(
                    yi, l, h,
                    linewidth=1.2,
                    alpha=0.18,
                    color=color_map.get(race, "#7f7f7f"),
                    zorder=2
                )

        if race == "White":
            ax.scatter(
                x, y,
                s=95,
                label=race,
                facecolors="none",
                edgecolors=color_map.get(race, "#7f7f7f"),
                linewidths=1.6,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(
                x, y,
                s=95,
                label=race,
                color=color_map.get(race, "#7f7f7f"),
                edgecolors="none",
                alpha=0.95,
                zorder=3,
            )

    for y in np.arange(-0.5, len(categories), 1):
        ax.axhline(y, color="gray", linewidth=0.8, alpha=0.35, zorder=1)

    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=11)
    ax.invert_yaxis()

    ax.set_xlabel("Wobbler Charged as Felony (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Charge Category", fontsize=12, fontweight="bold")
    ax.set_title(
        "Felony Charging Rate for Wobbler Charges by Race and Charge Category",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    xmin = np.nanmin(g["lo"].to_numpy(dtype=float))
    xmax = np.nanmax(g["hi"].to_numpy(dtype=float))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax.set_xlim(max(0, xmin - 2), min(100, xmax + 2))
    else:
        ax.set_xlim(0, 100)

    ax.legend(
        title="Canonical Race",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    return fig



def export_figure_to_pdf(fig, fig_name):
    """
    Export a single figure to PDF for Overleaf/LaTeX.
    """
    os.makedirs('../output/figures', exist_ok=True)
    output_path = f"../output/figures/{fig_name}.pdf"
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_name}.pdf")