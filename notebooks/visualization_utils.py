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

STATUTE_COLOR_MAP = {
    "Felony": "#E69F00", # Orange
    "Misdemeanor": "#F0E442" # Yellow
}

CLASSIFICATION_COLOR_MAP = {
    "Strict": "#E69F00", # Orange
    "Mixed": "#F0E442" # Yellow
}

STOP_TYPE_COLOR_MAP = {
    "Solo": "#E69F00", # Orange
    "Multiperson": "#F0E442" # Yellow
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



def create_sensitivity_visualization(strict_df, mixed_df, strict_name, mixed_name, title_ending):
    """
    Create publication-ready side-by-side comparison of strict vs. mixed search classifications.
    """

    # Filter to most recent year automatically
    latest_year = strict_df['Year'].max()
    strict = strict_df[strict_df['Year'] == latest_year].copy()
    mixed = mixed_df[mixed_df['Year'] == latest_year].copy()

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian"
    ]

    # Sort both dataframes by race order
    strict = strict.set_index("Perceived Race").loc[race_order].reset_index()
    mixed = mixed.set_index("Perceived Race").loc[race_order].reset_index()

    # Create figure with improved styling
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Sensitivity Analysis: {strict_name} vs. {mixed_name} {title_ending} ({latest_year})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    x = np.arange(len(strict)) * 1.5  # Increased spacing between races
    width = 0.35

    # Use centralized classification color map
    strict_color = CLASSIFICATION_COLOR_MAP['Strict']
    mixed_color = CLASSIFICATION_COLOR_MAP['Mixed']
    
    # Helper function to darken a color
    def darken_color(color, factor=0.7):
        """Darken a hex color by multiplying RGB values by factor"""
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        darkened = tuple(c * factor for c in rgb)
        return darkened

    # ---- Panel A: Search Rates ----
    strict_search = strict['Search Rate'] * 100
    mixed_search = mixed['Search Rate'] * 100
    
    # Calculate standard errors in percentage points
    strict_se = strict['Search Rate SE'] * 100 * 1.96  # 95% CI
    mixed_se = mixed['Search Rate SE'] * 100 * 1.96

    axes[0].bar(
        [i - width/2 for i in x],
        strict_search,
        width,
        label=f'{strict_name}',
        color=strict_color,
        alpha=0.8,
        yerr=strict_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.4, 'ecolor': darken_color(strict_color)}
    )

    axes[0].bar(
        [i + width/2 for i in x],
        mixed_search,
        width,
        label=f'{mixed_name}',
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.4, 'ecolor': darken_color(mixed_color)}
    )

    # Add value labels
    max_y_search = 0
    for i, idx in enumerate(x):
        s_val = strict_search.iloc[i]
        m_val = mixed_search.iloc[i]
        s_n = int(strict.iloc[i]['Search Count'])
        m_n = int(mixed.iloc[i]['Search Count'])
        s_se = strict_se.iloc[i]
        m_se = mixed_se.iloc[i]
        
        # Calculate exact x positions for bars
        x1 = idx - width/2
        x2 = idx + width/2
        
        # Value labels aligned with each bar
        axes[0].text(x1, s_val + s_se + 0.3,
                    f'{s_val:.1f}%\n(n={s_n:,})',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        axes[0].text(x2, m_val + m_se + 0.3,
                    f'{m_val:.1f}%\n(n={m_n:,})',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Update max y value
        max_y_search = max(max_y_search, 
                          s_val + s_se + 3,
                          m_val + m_se + 3)

    axes[0].set_title('(A) Conditional Search Rate', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_ylabel('Search Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Perceived Race', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strict['Perceived Race'], rotation=0, ha='center', fontsize=10)
    axes[0].legend(fontsize=10, frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle=':', axis='y')
    axes[0].set_axisbelow(True)
    # Tighter y-axis range
    axes[0].set_ylim(0, max_y_search * 1.02)

    # ---- Panel B: Hit Rates ----
    strict_hit = strict['Hit Rate'] * 100
    mixed_hit = mixed['Hit Rate'] * 100
    
    strict_hit_se = strict['Hit Rate SE'] * 100 * 1.96
    mixed_hit_se = mixed['Hit Rate SE'] * 100 * 1.96

    axes[1].bar(
        [i - width/2 for i in x],
        strict_hit,
        width,
        label=f'{strict_name}',
        color=strict_color,
        alpha=0.8,
        yerr=strict_hit_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.4, 'ecolor': darken_color(strict_color)}
    )

    axes[1].bar(
        [i + width/2 for i in x],
        mixed_hit,
        width,
        label=f'{mixed_name}',
        color=mixed_color,
        alpha=0.8,
        yerr=mixed_hit_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.4, 'ecolor': darken_color(mixed_color)}
    )

    # Add value labels
    max_y_hit = 0
    for i, idx in enumerate(x):
        s_val = strict_hit.iloc[i]
        m_val = mixed_hit.iloc[i]
        s_n = int(strict.iloc[i]['Hit Count'])
        m_n = int(mixed.iloc[i]['Hit Count'])
        s_se = strict_hit_se.iloc[i]
        m_se = mixed_hit_se.iloc[i]
        
        # Calculate exact x positions for bars
        x1 = idx - width/2
        x2 = idx + width/2
        
        # Value labels aligned with each bar
        axes[1].text(x1, s_val + s_se + 0.5,
                    f'{s_val:.1f}%\n(n={s_n:,})',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        axes[1].text(x2, m_val + m_se + 0.5,
                    f'{m_val:.1f}%\n(n={m_n:,})',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Update max y value
        max_y_hit = max(max_y_hit,
                       s_val + s_se + 5,
                       m_val + m_se + 5)

    axes[1].set_title('(B) Contraband Hit Rate', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Perceived Race', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mixed['Perceived Race'], rotation=0, ha='center', fontsize=10)
    axes[1].legend(fontsize=10, frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle=':', axis='y')
    axes[1].set_axisbelow(True)
    # Tighter y-axis range
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
    - y: charge category
    - x: felony filing rate (%)
    - points colored by race; horizontal 95% CI whiskers
    - races are "within category" via small vertical offsets (dodged)
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

    # Keep only races of interest, enforce order
    g = g[g["race_std"].isin(races)].copy()
    g["race_std"] = pd.Categorical(g["race_std"], categories=races, ordered=True)

    # Filter to categories with over 500 cases total
    cat_totals = g.groupby("charge_category")["Total"].sum()
    categories_over_500 = cat_totals[cat_totals > 500].index
    g = g[g["charge_category"].isin(categories_over_500)].copy()
    
    # Recalculate category totals after filtering
    cat_totals = g.groupby("charge_category")["Total"].sum().sort_values(ascending=False)

    overall_rate = g.groupby("charge_category")[["Felony", "Misdemeanor"]].sum()
    overall_rate["Total"] = overall_rate["Felony"] + overall_rate["Misdemeanor"]
    overall_rate["Rate"] = np.where(overall_rate["Total"] > 0, overall_rate["Felony"] / overall_rate["Total"], np.nan)

    # Choose which categories to plot (and in what order)
    if top_categories is not None:
        # Allow: list/tuple/Index/Series of category names
        categories = list(top_categories)

        # Keep only categories that actually exist after your filters
        existing = set(g["charge_category"].unique())
        categories = [c for c in categories if c in existing]
    else:
        # Fall back to automatic ordering logic
        if sort_by == "total":
            order = cat_totals.index.tolist()
        elif sort_by == "rate_overall":
            order = overall_rate["Rate"].sort_values(ascending=False).index.tolist()
        elif sort_by == "rate_black":
            blk = g[g["race_std"] == races[0]].set_index("charge_category")["Felony Rate"]
            order = blk.sort_values(ascending=False).index.tolist()
        else:  # "alpha" or fallback
            order = sorted(cat_totals.index.tolist())

        categories = order

    g = g[g["charge_category"].isin(categories)].copy()

    # Prepare y positions (one row per category) + dodges for race
    categories = list(categories)
    y_base = np.arange(len(categories))

    # small symmetric offsets so races sit "within category"
    offsets = np.linspace(-0.32, 0.32, num=len(races))

    # Figure sizing
    fig_h = max(6, 0.35 * len(categories) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # Light reference grid
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)

    # Alternating background bands for each category row
    for i, y0 in enumerate(y_base):
        if i % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", alpha=1.0, zorder=0)

    # Plot each race as a series
    for r_i, race in enumerate(races):
        sub = g[g["race_std"] == race].copy()
        # align to category list
        sub = sub.set_index("charge_category").reindex(categories)
        y = y_base + offsets[r_i]

        x = sub["x"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        n = sub["Total"].to_numpy(dtype=float)

        # CI whiskers
        for yi, xi, l, h, nn in zip(y, x, lo, hi, n):  # noqa: E741
            if np.isfinite(xi) and nn > 0 and np.isfinite(l) and np.isfinite(h):
                ax.hlines(yi, l, h, linewidth=1.2, alpha=0.18, color=color_map.get(race, "#7f7f7f"), zorder=2)

        # points
        if race == "White":
            ax.scatter(
                x, y,
                s=95,
                label=race,
                facecolors="none",                       # hollow fill
                edgecolors=color_map.get(race, "#7f7f7f"),# colored outline
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

    # Category separators aligned with shading edges
    for y in np.arange(-0.5, len(categories), 1):
        ax.axhline(y, color="gray", linewidth=0.8, alpha=0.35, zorder=1)
    
    # Y axis labels (category once)
    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=11)
    ax.invert_yaxis()  # common Cleveland style: top category first

    # X axis
    ax.set_xlabel("Wobbler Charged as Felony (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Charge Category", fontsize=12, fontweight="bold")
    ax.set_title(
        "Felony Charging Rate for Wobbler Charges by Race and Charge Category",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Keep bounds sane
    xmin = np.nanmin(g["lo"].to_numpy(dtype=float))
    xmax = np.nanmax(g["hi"].to_numpy(dtype=float))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax.set_xlim(max(0, xmin - 2), min(100, xmax + 2))
    else:
        ax.set_xlim(0, 100)

    # Legend
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