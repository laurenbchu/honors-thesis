import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

from matplotlib.ticker import FuncFormatter

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



def visualize_policing(policing_analysis):

    df = policing_analysis.copy()
    race_col = "Perceived Race"
    
    from table_utils import visualization_setup
    df = visualization_setup(df, race_col=race_col)
    
    # Use centralized color map
    color_map = COLOR_MAP
    
    years = sorted(df["Year"].dropna().unique().tolist())
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), gridspec_kw={'wspace': 0.25, 'hspace': 0.30})
    axes = axes.flatten()
    
    # Define the four panels
    panels = [
        {
            'ax_idx': 0,
            'y_col': 'Stops per 1,000',
            'title': '(A) Police Stop Rates\n(per 1,000 residents)',
            'ylabel': 'Stops per 1,000 Residents',
            'count_col': 'Stop Count',
            'is_rate': False
        },
        {
            'ax_idx': 1,
            'y_col': 'Searches per 1,000',
            'title': '(B) Searches per Capita\n(per 1,000 residents)',
            'ylabel': 'Searches per 1,000 Residents',
            'count_col': 'Search Count',
            'is_rate': False
        },
        {
            'ax_idx': 2,
            'y_col': 'Search Rate',
            'title': '(C) Conditional Search Rate\n(among those stopped)',
            'ylabel': 'Search Rate',
            'count_col': 'Search Count',
            'is_rate': True
        },
        {
            'ax_idx': 3,
            'y_col': 'Hit Rate',
            'title': '(D) Contraband Hit Rate\n(contraband found given search)',
            'ylabel': 'Hit Rate',
            'count_col': 'Hit Count',
            'is_rate': True
        }
    ]
    
    for panel in panels:
        ax = axes[panel['ax_idx']]
        y_col = panel['y_col']
        count_col = panel['count_col']
        
        for race in color_map.keys():
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                
                x_vals = d["Year"]
                
                # Plot main line
                ax.plot(x_vals, d[y_col], 
                       marker="o", markersize=8,
                       linewidth=linewidth, linestyle=linestyle,
                       color=color_map[race], label=race)
                
                # Add error ribbons if available
                se_col = f"{y_col} SE"
                if se_col in d.columns and d[se_col].notna().any():
                    ci_lower = d[y_col] - d[se_col]*1.96
                    ci_upper = d[y_col] + d[se_col]*1.96
                    
                    ribbon_alpha = 0.12 if "Hit Rate" in y_col else 0.2
                    ax.fill_between(x_vals, ci_lower, ci_upper,
                                   color=color_map[race], 
                                   alpha=ribbon_alpha, linewidth=0)
                
                # Add sample size annotations for 2024 only
                if count_col and count_col in d.columns:
                    for idx, row in d.iterrows():
                        if row["Year"] == 2024:
                            n = int(row[count_col])
                            x_pos = row["Year"]
                            y_val = row[y_col]
                            
                            ax.annotate(f'n={n:,}', 
                                      xy=(x_pos, y_val),
                                      xytext=(0, 12), textcoords='offset points',
                                      fontsize=8, ha='center', alpha=0.9,
                                      color=color_map[race],
                                      bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='white', 
                                               edgecolor='none',
                                               alpha=0.7))
        
        ax.set_xlabel("Year", fontsize=11, fontweight='bold')
        ax.set_ylabel(panel['ylabel'], fontsize=11, fontweight='bold')
        ax.set_title(panel['title'], fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # Format y-axis
        def clean_half_steps(y, _):
            return f'{int(y)}' if float(y).is_integer() else f'{y:.1f}'

        if panel['y_col'] == 'Stops per 1,000':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))
        elif panel['y_col'] == 'Searches per 1,000':
            ax.yaxis.set_major_formatter(FuncFormatter(clean_half_steps))
        elif panel['is_rate']:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add extra space at top for labels
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + 0.15 * y_range)
    
    # Add overall title
    fig.suptitle(
        'Police Contact Patterns by Perceived Race',
        fontsize=16,
        fontweight='bold',
        y=0.965
    )

    # Add centered legend between title and plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=4
    )

    # Manually reserve space for title + legend
    fig.subplots_adjust(
        top=0.87,
        bottom=0.08,
        left=0.07,
        right=0.98,
        hspace=0.30,
        wspace=0.25
    )

    return fig



def visualize_search_and_hit_rates_by_reason(policing_by_reason):
    """
    Create a 1x2 figure of pooled (2022-2024) search and hit rates
    by reason for contact and perceived race.

    Panel A: Search rate by reason for contact
    Panel B: Hit rate by reason for contact
    """

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian"]
    reason_order = list(policing_by_reason.keys())

    # Build one pooled dataframe across all reasons
    pooled_rows = []

    for reason, rates_df in policing_by_reason.items():
        df = rates_df[rates_df["Perceived Race"] != "Other"].copy()

        pooled = (
            df.groupby("Perceived Race", as_index=False)
            .agg({
                "Stop Count": "sum",
                "Search Count": "sum",
                "Hit Count": "sum"
            })
        )

        pooled["Reason"] = reason

        # Search rate and SE
        pooled["Search Rate"] = np.where(
            pooled["Stop Count"] > 0,
            pooled["Search Count"] / pooled["Stop Count"],
            np.nan
        )
        pooled["Search Rate SE"] = np.where(
            pooled["Stop Count"] > 0,
            np.sqrt(pooled["Search Rate"] * (1 - pooled["Search Rate"]) / pooled["Stop Count"]),
            np.nan
        )

        # Hit rate and SE
        pooled["Hit Rate"] = np.where(
            pooled["Search Count"] > 0,
            pooled["Hit Count"] / pooled["Search Count"],
            np.nan
        )
        pooled["Hit Rate SE"] = np.where(
            pooled["Search Count"] > 0,
            np.sqrt(pooled["Hit Rate"] * (1 - pooled["Hit Rate"]) / pooled["Search Count"]),
            np.nan
        )

        pooled_rows.append(pooled)

    pooled_all = pd.concat(pooled_rows, ignore_index=True)

    # Enforce consistent ordering
    pooled_all["Perceived Race"] = pd.Categorical(
        pooled_all["Perceived Race"],
        categories=race_order,
        ordered=True
    )
    pooled_all["Reason"] = pd.Categorical(
        pooled_all["Reason"],
        categories=reason_order,
        ordered=True
    )
    pooled_all = pooled_all.sort_values(["Perceived Race", "Reason"])

    # Convert to percent + 95% CI
    pooled_all["Search Rate %"] = pooled_all["Search Rate"] * 100
    pooled_all["Search CI %"] = pooled_all["Search Rate SE"] * 1.96 * 100
    pooled_all["Hit Rate %"] = pooled_all["Hit Rate"] * 100
    pooled_all["Hit CI %"] = pooled_all["Hit Rate SE"] * 1.96 * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"wspace": 0.18})
    x = np.arange(len(reason_order))

    # Small offsets so races do not sit exactly on top of each other
    offsets = {
        "Black/African American": -0.18,
        "Hispanic/Latino": -0.06,
        "White": 0.06,
        "Asian": 0.18
    }

    panel_specs = [
        {
            "ax": axes[0],
            "y_col": "Search Rate %",
            "ci_col": "Search CI %",
            "ylabel": "Search Rate (%)",
            "title": "(A) Search Rate by Reason for Contact"
        },
        {
            "ax": axes[1],
            "y_col": "Hit Rate %",
            "ci_col": "Hit CI %",
            "ylabel": "Hit Rate (%)",
            "title": "(B) Hit Rate by Reason for Contact"
        }
    ]

    for panel in panel_specs:
        ax = panel["ax"]

        for race in race_order:
            d = pooled_all[pooled_all["Perceived Race"] == race].sort_values("Reason")
            if d.empty:
                continue

            y = d[panel["y_col"]].to_numpy()
            yerr = d[panel["ci_col"]].to_numpy()
            x_race = x + offsets[race]

            linewidth = 2.5 if race == "White" else 2
            linestyle = "--" if race == "White" else "-"
            marker = "o"

            ax.errorbar(
            x_race,
            y,
            yerr=yerr,
            fmt='o',
            linestyle='none',   # no connecting lines
            markersize=8,
            capsize=4,
            capthick=1.5,
            elinewidth=1.8,
            color=COLOR_MAP[race],
            label=race
        )

        ax.set_title(panel["title"], fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Reason for Contact", fontsize=11, fontweight="bold")
        ax.set_ylabel(panel["ylabel"], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(reason_order, rotation=0, ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

        # Give a little headroom
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(max(0, y_min), y_max * 1.08)

    fig.suptitle(
        "Search and Contraband Hit Rates by Reason for Contact and Race (2022–2024 Pooled)",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=True,
        fontsize=11
    )

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98)

    return fig



def plot_agency_black_white_hit_rates(agency_hit_df):
    """
    Create a publication-ready scatter plot comparing White and Black hit rates by agency.
    
    Points above the diagonal (higher Black hit rate) are colored green.
    Points below the diagonal (higher White hit rate) are colored red.
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



def create_combined_sensitivity_visualization(baseline_df, mixed_df, multiperson_df):
    """
    Create a publication-ready two-panel sensitivity figure comparing:
    - Baseline
    - Mixed classification
    - Multiperson stops

    Panel A: Conditional Search Rate
    Panel B: Contraband Hit Rate
    """

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
    Returns a dict with:
      - 'assault_violence_weapons': 2x2 figure
          Row 1: Assault/Violence (Misdemeanor, Felony)
          Row 2: Weapons (Misdemeanor, Felony)
      - 'dui': 1x2 figure
          DUI (Misdemeanor, Felony)

    Bars are colored by race using COLOR_MAP and include 95% CI error bars.
    """

    df = enhancement_by_primary.copy()

    # --- resolve category names exactly as they appear in the data ---
    available_categories = df["primary_charge_category"].dropna().astype(str).unique().tolist()

    def resolve_category_name(target):
        target_norm = target.strip().lower()
        for cat in available_categories:
            if cat.strip().lower() == target_norm:
                return cat
        raise ValueError(
            f"Could not find category '{target}' in primary_charge_category.\n"
            f"Available categories: {sorted(available_categories)}"
        )

    assault_cat = resolve_category_name("Assault/Violence")
    weapons_cat = resolve_category_name("Weapons")
    dui_cat = resolve_category_name("DUI")

    # Optional filtering if user passes a subset
    if top_categories is not None:
        keep = {str(x).strip().lower() for x in top_categories}
        requested = {
            assault_cat.strip().lower(),
            weapons_cat.strip().lower(),
            dui_cat.strip().lower()
        }
        use_requested = requested.intersection(keep)
        if use_requested:
            df = df[df["primary_charge_category"].str.strip().str.lower().isin(use_requested)].copy()

    def darken_color(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)

    def summarize_category(category_name):
        cat = df[df["primary_charge_category"] == category_name].copy()

        summary = (
            cat.groupby(["race_std", "primary_statute_level"], as_index=False)
               .agg({"Enhanced": "sum", "N": "sum"})
        )

        if summary.empty:
            return None, None

        summary["Enhancement Rate"] = np.where(
            summary["N"] > 0,
            summary["Enhanced"] / summary["N"],
            np.nan
        )
        summary["SE"] = np.where(
            summary["N"] > 0,
            np.sqrt(summary["Enhancement Rate"] * (1 - summary["Enhancement Rate"]) / summary["N"]),
            np.nan
        )

        summary["race_std"] = pd.Categorical(
            summary["race_std"],
            categories=RACE_ORDER,
            ordered=True
        )
        summary = summary.sort_values(["race_std", "primary_statute_level"])

        present_races = [
            r for r in RACE_ORDER
            if r in set(summary["race_std"].dropna().astype(str))
        ]

        return summary, present_races

    def aligned_arrays(summary, races, statute_level):
        sub = summary[summary["primary_statute_level"] == statute_level].copy()
        lookup = {str(row["race_std"]): row for _, row in sub.iterrows()}

        rates = np.array([
            100 * lookup[r]["Enhancement Rate"] if r in lookup else np.nan
            for r in races
        ], dtype=float)

        ses = np.array([
            100 * lookup[r]["SE"] if r in lookup else np.nan
            for r in races
        ], dtype=float)

        ns = np.array([
            lookup[r]["N"] if r in lookup else 0
            for r in races
        ], dtype=float)

        errs = ses * Z
        return rates, errs, ns

    def plot_panel(ax, rates, errs, ns, races, panel_title, show_ylabel=False):
        x = np.arange(len(races))
        colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]

        heights = np.where(np.isfinite(rates), rates, 0.0)
        bars = ax.bar(
            x,
            heights,
            color=colors,
            alpha=0.85,
            edgecolor="none",
            linewidth=0
        )

        # Hide truly missing bars rather than showing a visible zero bar
        for bar, rate in zip(bars, rates):
            if not np.isfinite(rate):
                bar.set_alpha(0.0)

        for i, (rate, err, col, n) in enumerate(zip(rates, errs, colors, ns)):
            if np.isfinite(rate) and n > 0:
                ax.errorbar(
                    i, rate,
                    yerr=err,
                    fmt="none",
                    ecolor=darken_color(col),
                    alpha=0.4,
                    capsize=5,
                    capthick=1.5,
                    linewidth=1.5
                )

        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Canonical Race", fontsize=12, fontweight="bold")
        if show_ylabel:
            ax.set_ylabel("Enhancement Rate (%)", fontsize=12, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=0, ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)

        max_y = 0
        for i, (rate, err, n) in enumerate(zip(rates, errs, ns)):
            if np.isfinite(rate) and n > 0:
                y_pos = rate + err + 0.3
                ax.text(
                    i, y_pos,
                    f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )
                max_y = max(max_y, y_pos + 2)

        return max_y

    def make_two_panel_category_figure(category_name, figure_title):
        summary, races = summarize_category(category_name)
        if summary is None or not races:
            return None

        mis_rates, mis_errs, mis_n = aligned_arrays(summary, races, "Misdemeanor")
        fel_rates, fel_errs, fel_n = aligned_arrays(summary, races, "Felony")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"wspace": 0.3})

        max_y1 = plot_panel(
            ax1, mis_rates, mis_errs, mis_n, races,
            "(A) Misdemeanor Charges",
            show_ylabel=True
        )
        max_y2 = plot_panel(
            ax2, fel_rates, fel_errs, fel_n, races,
            "(B) Felony Charges",
            show_ylabel=False
        )

        ymax = min(100, max(max_y1, max_y2) + 2 if max(max_y1, max_y2) > 0 else 100)
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)

        fig.suptitle(
            figure_title,
            fontsize=15,
            fontweight="bold",
            y=0.98
        )
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.17, wspace=0.3)

        return fig

    def make_four_panel_combined_figure(cat1, cat2, figure_title):
        summary1, races1 = summarize_category(cat1)
        summary2, races2 = summarize_category(cat2)

        if summary1 is None and summary2 is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(18, 13), gridspec_kw={"wspace": 0.3, "hspace": 0.4})
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Row 1: Assault/Violence
        if summary1 is not None and races1:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary1, races1, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary1, races1, "Felony")

            max_y1 = plot_panel(
                ax1, mis_rates, mis_errs, mis_n, races1,
                f"(A) {cat1}: Misdemeanor",
                show_ylabel=True
            )
            max_y2 = plot_panel(
                ax2, fel_rates, fel_errs, fel_n, races1,
                f"(B) {cat1}: Felony",
                show_ylabel=False
            )

            ymax_row1 = min(100, max(max_y1, max_y2) + 2 if max(max_y1, max_y2) > 0 else 100)
            ax1.set_ylim(0, ymax_row1)
            ax2.set_ylim(0, ymax_row1)
        else:
            ax1.axis("off")
            ax2.axis("off")

        # Row 2: Weapons
        if summary2 is not None and races2:
            mis_rates, mis_errs, mis_n = aligned_arrays(summary2, races2, "Misdemeanor")
            fel_rates, fel_errs, fel_n = aligned_arrays(summary2, races2, "Felony")

            max_y3 = plot_panel(
                ax3, mis_rates, mis_errs, mis_n, races2,
                f"(C) {cat2}: Misdemeanor",
                show_ylabel=True
            )
            max_y4 = plot_panel(
                ax4, fel_rates, fel_errs, fel_n, races2,
                f"(D) {cat2}: Felony",
                show_ylabel=False
            )

            ymax_row2 = min(100, max(max_y3, max_y4) + 2 if max(max_y3, max_y4) > 0 else 100)
            ax3.set_ylim(0, ymax_row2)
            ax4.set_ylim(0, ymax_row2)
        else:
            ax3.axis("off")
            ax4.axis("off")

        fig.suptitle(
            figure_title,
            fontsize=16,
            fontweight="bold",
            y=0.98
        )
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.12, wspace=0.3, hspace=0.4)

        return fig

    figs = {}

    figs["assault_violence_weapons"] = make_four_panel_combined_figure(
        assault_cat,
        weapons_cat,
        "Enhancement Charge Rate by Race and Statute Level:\nAssault/Violence and Weapons"
    )

    figs["dui"] = make_two_panel_category_figure(
        dui_cat,
        "Enhancement Charge Rate by Race and Statute Level:\nDUI"
    )

    return figs



def plot_wobbler_combined(df, top_categories=None, sort_by="rate_overall"):
    """
    Create a combined two-panel figure for wobbler charges:
    - Panel A: Overall wobbler felony filing rate by race
    - Panel B: Cleveland dot plot by charge category and race
    """
    
    Z_ = globals().get("Z", 1.96)
    races = list(globals().get("RACE_ORDER", ["Black/African American", "Hispanic/Latino", "White", "Asian"]))
    color_map = globals().get("COLOR_MAP", {})
    
    # Helper: darken a color for error bars
    def darken(color, factor=0.7):
        rgb = mcolors.to_rgb(color)
        return tuple(c * factor for c in rgb)
    
    # Calculate figure height based on number of categories for Panel B
    wobblers = df[df["is_wobbler"]].copy()
    
    # Determine categories for Panel B
    if top_categories is not None:
        g = wobblers.groupby(["charge_category", "race_std", "statute_level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Felony", "Misdemeanor"]:
            if col not in g.columns:
                g[col] = 0
        g["Total"] = g["Felony"] + g["Misdemeanor"]
        existing = set(g["charge_category"].unique())
        categories = [c for c in top_categories if c in existing]
        n_categories = len(categories)
    else:
        g = wobblers.groupby(["charge_category", "race_std", "statute_level"]).size().unstack(fill_value=0).reset_index()
        for col in ["Felony", "Misdemeanor"]:
            if col not in g.columns:
                g[col] = 0
        g["Total"] = g["Felony"] + g["Misdemeanor"]
        n_categories = len(g["charge_category"].unique())
    
    # Create figure with side-by-side panels
    fig_height = max(8, 0.35 * n_categories + 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, fig_height), gridspec_kw={'wspace': 0.30, 'width_ratios': [1, 1.5]})
    
    # ==========================================
    # PANEL A: Overall Wobbler Felony Rate
    # ==========================================
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
    
    summary = summary.reindex(races)
    
    races_a = summary.index.astype(str).tolist()
    rates = (summary["Felony Rate"].to_numpy(dtype=float) * 100)
    ses = (summary["Felony Rate SE"].to_numpy(dtype=float) * 100)
    ns = summary["Total"].to_numpy(dtype=float)
    errs = ses * Z_
    
    colors = [color_map.get(r, "#7f7f7f") for r in races_a]
    x = np.arange(len(races_a))
    
    ax1.bar(x, np.nan_to_num(rates, nan=0.0), color=colors, alpha=0.85, edgecolor="none", linewidth=0)
    
    for i, (rate, err, col, n) in enumerate(zip(rates, errs, colors, ns)):
        if np.isfinite(rate) and n > 0:
            ax1.errorbar(i, rate, yerr=err, fmt="none", ecolor=darken(col),
                        alpha=0.4, capsize=5, capthick=1.5, linewidth=1.5)
    
    ax1.set_title("(A) Overall Wobbler Felony Filing Rate by Race",
                 fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Canonical Race", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Wobbler Charged as Felony (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(races_a, fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax1.set_axisbelow(True)
    
    max_y = 0
    for i, (rate, err, n) in enumerate(zip(rates, errs, ns)):
        if np.isfinite(rate) and n > 0:
            y_pos = rate + err + 0.7
            ax1.text(i, y_pos, f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            max_y = max(max_y, y_pos + 2)
    
    pad = 2
    ymax = max_y if max_y > 0 else 100
    ax1.set_ylim(0, min(100, ymax + pad))
    
    # ==========================================
    # PANEL B: Cleveland Dot Plot by Category
    # ==========================================
    g = (
        wobblers.groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    
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
    
    g = g[g["race_std"].isin(races)].copy()
    g["race_std"] = pd.Categorical(g["race_std"], categories=races, ordered=True)
    
    # Determine and sort categories
    if top_categories is not None:
        categories = list(top_categories)
        existing = set(g["charge_category"].unique())
        categories = [c for c in categories if c in existing]
        g = g[g["charge_category"].isin(categories)].copy()
    else:
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
    
    g = g[g["charge_category"].isin(categories)].copy()
    categories = list(categories)
    
    y_base = np.arange(len(categories))
    offsets = np.linspace(-0.32, 0.32, num=len(races))
    
    ax2.grid(axis="x", alpha=0.15)
    ax2.set_axisbelow(True)
    
    for i, y0 in enumerate(y_base):
        if i % 2 == 0:
            ax2.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", alpha=1.0, zorder=0)
    
    for r_i, race in enumerate(races):
        sub = g[g["race_std"] == race].copy()
        sub = sub.set_index("charge_category").reindex(categories)
        y = y_base + offsets[r_i]
        
        x_vals = sub["x"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        n = sub["Total"].to_numpy(dtype=float)
        
        for yi, xi, l, h, nn in zip(y, x_vals, lo, hi, n):
            if np.isfinite(xi) and nn > 0 and np.isfinite(l) and np.isfinite(h):
                ax2.hlines(
                    yi, l, h,
                    linewidth=1.2,
                    alpha=0.18,
                    color=color_map.get(race, "#7f7f7f"),
                    zorder=2
                )
        
        if race == "White":
            ax2.scatter(
                x_vals, y,
                s=95,
                label=race,
                facecolors="none",
                edgecolors=color_map.get(race, "#7f7f7f"),
                linewidths=1.6,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax2.scatter(
                x_vals, y,
                s=95,
                label=race,
                color=color_map.get(race, "#7f7f7f"),
                edgecolors="none",
                alpha=0.95,
                zorder=3,
            )
    
    for y in np.arange(-0.5, len(categories), 1):
        ax2.axhline(y, color="gray", linewidth=0.8, alpha=0.35, zorder=1)
    
    ax2.set_yticks(y_base)
    ax2.set_yticklabels(categories, fontsize=11)
    ax2.invert_yaxis()
    
    ax2.set_xlabel("Wobbler Charged as Felony (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Charge Category", fontsize=11, fontweight="bold")
    ax2.set_title(
        "(B) Felony Charging Rate by Charge Category and Race",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    
    xmin = np.nanmin(g["lo"].to_numpy(dtype=float))
    xmax = np.nanmax(g["hi"].to_numpy(dtype=float))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax2.set_xlim(max(0, xmin - 2), min(100, xmax + 2))
    else:
        ax2.set_xlim(0, 100)
    
    ax2.legend(
        title="Canonical Race",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        loc="upper left"
    )
    
    # Overall title
    fig.suptitle(
        "Wobbler Charges: Felony Filing Rates by Race and Charge Category\n(with 95% Confidence Intervals)",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])

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