import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def fmt_est_ci(est, se, digits=2):
    """Format like: 189.86 (±11.38)"""
    return f"{est:.{digits}f} (±{(1.96*se):.{digits}f})"

def make_pivot_with_ci(df, value_col, se_col):
    """
    Returns a wide table with one column per year containing formatted estimate (±CI).
    """
    tmp = df[["Perceived Race", "Year", value_col, se_col]].copy()
    tmp["cell"] = [fmt_est_ci(e, s, digits=2) for e, s in zip(tmp[value_col], tmp[se_col])]
    wide = tmp.pivot(index="Perceived Race", columns="Year", values="cell")
    # order columns if desired
    wide = wide.reindex(columns=[y for y in (2022, 2023, 2024) if y in wide.columns])
    return wide

def visualization_setup(df, race_col):
    """
    Enforce a consistent race order and sort for plotting.
    race_col should be either:
      - "Perceived Race" (policing)
      - "Canonical Race" (prosecution)
    """
    out = df.copy()

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian"]

    if race_col in out.columns:
        out[race_col] = pd.Categorical(out[race_col], categories=race_order, ordered=True)

    if "Year" in out.columns and race_col in out.columns:
        out = out.sort_values(["Year", race_col])

    return out

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

def _binom_ci(enhanced, n, z=1.96):
    """
    Normal-approx binomial CI (Wald), clipped to [0,1].
    Returns p, se, lo, hi as numpy arrays.
    """
    enhanced = np.asarray(enhanced, dtype=float)
    n = np.asarray(n, dtype=float)

    p = np.where(n > 0, enhanced / n, np.nan)
    se = np.where(n > 0, np.sqrt(p * (1 - p) / n), np.nan)
    lo = np.maximum(0, p - z * se)
    hi = np.minimum(1, p + z * se)
    return p, se, lo, hi


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

def export_figures_to_pdf(figs_dict, output_dir='../output'):
    """
    Export all figures to PDF format for high-quality inclusion in Overleaf/LaTeX documents.
    
    Parameters:
    -----------
    figs_dict : dict
        Dictionary of figures from visualize_policing() or visualize_prosecution()
        Keys are figure names, values are matplotlib figure objects
    output_dir : str
        Directory to save PDF files (default: '../output')
    
    Example:
    --------
    policing_figs = visualize_policing(policing_analysis)
    export_figures_to_pdf(policing_figs)
    
    prosecution_figs = visualize_prosecution(prosecution_analysis)
    export_figures_to_pdf(prosecution_figs)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fig_name, fig in figs_dict.items():
        output_path = f'{output_dir}/{fig_name}.pdf'
        fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    print(f"\nAll figures exported to {output_dir}/ as PDF files")

# ------------------------------------------------------------------
# Policing
# ------------------------------------------------------------------

def visualize_policing(policing_analysis):
    df = policing_analysis.copy()
    race_col = "Perceived Race"
    
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
                        
                        # Create semi-transparent ribbon
                        ax.fill_between(x_vals, ci_lower, ci_upper,
                                       color=color_map[race], 
                                       alpha=0.2, linewidth=0)
                
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
                 frameon=True, fancybox=True, shadow=True, 
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
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
                                    color=color_map[race], alpha=0.2, linewidth=0)
                
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
                     fontsize=13, fontweight='bold', pad=15)
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
                                    color=color_map[race], alpha=0.2, linewidth=0)
                
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
                     fontsize=13, fontweight='bold', pad=15)
        # Legend will be added at figure level below
        ax2.set_xticks(years)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add extra space at top for labels
        y_min, y_max = ax2.get_ylim()
        y_range = y_max - y_min
        ax2.set_ylim(y_min, y_max + 0.15 * y_range)
        
        fig.suptitle('Police Search Patterns by Perceived Race', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add centered legend below the plots
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, title="Perceived Race", 
                  fontsize=10, title_fontsize=11,
                  frameon=True, fancybox=True, shadow=True,
                  loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
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
    
    Parameters
    ----------
    strict_df : pandas.DataFrame
        Policing analysis with strict discretionary classification
    mixed_df : pandas.DataFrame
        Policing analysis with mixed discretionary classification
        
    Returns
    -------
    matplotlib.figure.Figure
        Publication-ready comparison figure
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

    x = range(len(strict))
    width = 0.4  # Increased from 0.35 to widen bars

    # Use centralized classification color map
    strict_color = CLASSIFICATION_COLOR_MAP['Strict']
    mixed_color = CLASSIFICATION_COLOR_MAP['Mixed']

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
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
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
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
    )

    # Add value labels on bars with sample sizes
    max_y_search = 0  # Track maximum y value for axis limit
    for i, (s_val, m_val) in enumerate(zip(strict_search, mixed_search)):
        s_n = int(strict.iloc[i]['Search Count'])
        m_n = int(mixed.iloc[i]['Search Count'])
        
        axes[0].text(i - width/2, s_val + strict_se.iloc[i] + 0.5,
                    f'{s_val:.1f}%\n(n={s_n:,})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[0].text(i + width/2, m_val + mixed_se.iloc[i] + 0.5,
                    f'{m_val:.1f}%\n(n={m_n:,})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Update max y value (bar height + error bar + annotation space)
        max_y_search = max(max_y_search, 
                          s_val + strict_se.iloc[i] + 3,
                          m_val + mixed_se.iloc[i] + 3)

    axes[0].set_title('(A) Conditional Search Rate', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_ylabel('Search Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Perceived Race', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strict['Perceived Race'], rotation=45, ha='right', fontsize=10)
    axes[0].legend(fontsize=10, frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3, linestyle=':', axis='y')
    axes[0].set_axisbelow(True)
    # Automatically adjust y-axis limit to prevent label clipping
    axes[0].set_ylim(0, max_y_search)

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
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
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
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
    )

    # Add value labels on bars with sample sizes
    max_y_hit = 0  # Track maximum y value for axis limit
    for i, (s_val, m_val) in enumerate(zip(strict_hit, mixed_hit)):
        s_n = int(strict.iloc[i]['Hit Count'])
        m_n = int(mixed.iloc[i]['Hit Count'])
        
        axes[1].text(i - width/2, s_val + strict_hit_se.iloc[i] + 1,
                    f'{s_val:.1f}%\n(n={s_n:,})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1].text(i + width/2, m_val + mixed_hit_se.iloc[i] + 1,
                    f'{m_val:.1f}%\n(n={m_n:,})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Update max y value (bar height + error bar + annotation space)
        max_y_hit = max(max_y_hit, 
                       s_val + strict_hit_se.iloc[i] + 5,
                       m_val + mixed_hit_se.iloc[i] + 5)

    axes[1].set_title('(B) Contraband Hit Rate', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Perceived Race', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mixed['Perceived Race'], rotation=45, ha='right', fontsize=10)
    axes[1].legend(fontsize=10, frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3, linestyle=':', axis='y')
    axes[1].set_axisbelow(True)
    # Automatically adjust y-axis limit to prevent label clipping
    axes[1].set_ylim(0, max_y_hit)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig 
    
# ------------------------------------------------------------------
# Prosecution: Enhancement Rates
# ------------------------------------------------------------------

def enhancement_rate_race_table(enhancement_by_primary):
    """
    Overall enhancement rates by race (aggregating across all statute levels and charge categories).
    Returns a simple table with columns: Canonical Race, Number of Cases, Enhancement Rate (%).
    """
    summary = (
        enhancement_by_primary
        .groupby('race_std', as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[['race_std', 'N', 'Enhancement Rate (%)']].copy()
    result.columns = ['Canonical Race', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by race order
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result = result.sort_values('race_sort').drop('race_sort', axis=1)
    
    return result.reset_index(drop=True)


def enhancement_rate_race_statute_table(enhancement_by_primary):
    """
    Enhancement rates by race, stratified by statute level.
    Returns a long-format table with columns: Canonical Race, Statute Level, Number of Cases, Enhancement Rate (%).
    """
    summary = (
        enhancement_by_primary
        .groupby(['race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI using the standard format
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[['race_std', 'primary_statute_level', 'N', 'Enhancement Rate (%)']].copy()
    result.columns = ['Canonical Race', 'Statute Level', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by race order and statute level
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Misdemeanor", "Felony"]
    
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result['statute_sort'] = result['Statute Level'].map({stat: i for i, stat in enumerate(statute_order)})
    result = result.sort_values(['race_sort', 'statute_sort']).drop(['race_sort', 'statute_sort'], axis=1)
    
    # Blank out duplicate race names (show race only once per group)
    result['Canonical Race'] = result['Canonical Race'].mask(
        result['Canonical Race'].eq(result['Canonical Race'].shift()), ''
    )
    
    return result.reset_index(drop=True)


def enhancement_rate_race_statute_category_table(enhancement_by_primary, top_categories=None):
    """
    Enhancement rates by race, stratified by both statute level AND charge category.
    Returns a long-format table with columns: Charge Category, Canonical Race, Statute Level, Number of Cases, Enhancement Rate (%).
    
    Parameters
    ----------
    enhancement_by_primary : pd.DataFrame
        Enhancement data with columns: race_std, primary_statute_level, primary_charge_category, Enhanced, N
    top_categories : list, optional
        List of charge categories to include. If None, includes all categories.
    """
    df = enhancement_by_primary.copy()
    
    # Filter to specific categories if provided
    if top_categories is not None:
        df = df[df['primary_charge_category'].isin(top_categories)]
    
    summary = (
        df.groupby(['primary_charge_category', 'race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[[
        'primary_charge_category', 'race_std', 'primary_statute_level', 'N', 'Enhancement Rate (%)'
    ]].copy()
    result.columns = ['Charge Category', 'Canonical Race', 'Statute Level', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by category (by total N), then race, then statute level
    category_order = (
        summary.groupby('primary_charge_category')['N']
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Misdemeanor", "Felony"]
    
    result['cat_sort'] = result['Charge Category'].map({cat: i for i, cat in enumerate(category_order)})
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result['statute_sort'] = result['Statute Level'].map({stat: i for i, stat in enumerate(statute_order)})
    result = result.sort_values(['cat_sort', 'race_sort', 'statute_sort']).drop(['cat_sort', 'race_sort', 'statute_sort'], axis=1)
    
    # Blank out duplicate charge category and race names for readability
    result['Charge Category'] = result['Charge Category'].mask(
        result['Charge Category'].eq(result['Charge Category'].shift()), ''
    )
    result['Canonical Race'] = result['Canonical Race'].mask(
        result['Canonical Race'].eq(result['Canonical Race'].shift()), ''
    )
    
    return result.reset_index(drop=True)



def plot_enhancement_rate_by_race(enhancement_by_primary):
    """
    Plot 1: Overall enhancement rates by race (aggregated across all statute levels and categories).
    
    Parameters
    ----------
    enhancement_by_primary : pd.DataFrame
        Enhancement data with columns: race_std, Enhanced, N
        
    Returns
    -------
    matplotlib.figure.Figure
        Bar chart showing enhancement rates by race with 95% CI
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(races))
    colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]
    
    # Create bars
    bars = ax.bar(
        x, enh_rates,
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add error bars
    ax.errorbar(
        x, enh_rates,
        yerr=errors,
        fmt='none',
        ecolor='black',
        alpha=0.55,
        capsize=5,
        capthick=1.5,
        linewidth=1.5
    )
    
    # Set labels and title
    ax.set_ylabel('Enhancement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_title(
        'Enhancement Charge Rate by Race\n(Overall, with 95% Confidence Intervals)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add sample sizes and rates as text above bars
    for i, (rate, err, n) in enumerate(zip(enh_rates, errors, ns)):
        if np.isfinite(rate):
            ax.text(
                i, rate + err + 1,
                f'{rate:.1f}%\n(n={int(n):,})',
                ha='center',
                va='bottom',
                fontsize=9,
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
    Plot 2: Enhancement rates by race, stratified by statute level (Felony vs Misdemeanor).
    
    Parameters
    ----------
    enhancement_by_primary : pd.DataFrame
        Enhancement data with columns: race_std, primary_statute_level, Enhanced, N
        
    Returns
    -------
    matplotlib.figure.Figure
        Grouped bar chart showing enhancement rates by race and statute level with 95% CI
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
    
    # Extract data
    races = mis['race_std'].astype(str).to_list()
    mis_rates = mis['Enhancement Rate'].to_numpy() * 100
    fel_rates = fel['Enhancement Rate'].to_numpy() * 100
    mis_se = mis['SE'].to_numpy() * 100
    fel_se = fel['SE'].to_numpy() * 100
    mis_n = mis['N'].to_numpy()
    fel_n = fel['N'].to_numpy()
    
    # Calculate 95% CI
    mis_errors = mis_se * Z
    fel_errors = fel_se * Z
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(races))
    width = 0.35
    
    # Use centralized statute color map
    mis_color = STATUTE_COLOR_MAP['Misdemeanor']
    fel_color = STATUTE_COLOR_MAP['Felony']
    
    # Create grouped bars
    bars1 = ax.bar(
        x - width/2, mis_rates,
        width,
        label='Misdemeanor',
        color=mis_color,
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    
    bars2 = ax.bar(
        x + width/2, fel_rates,
        width,
        label='Felony',
        color=fel_color,
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add error bars
    ax.errorbar(
        x - width/2, mis_rates,
        yerr=mis_errors,
        fmt='none',
        ecolor='black',
        alpha=0.55,
        capsize=4,
        capthick=1.2,
        linewidth=1.2
    )
    
    ax.errorbar(
        x + width/2, fel_rates,
        yerr=fel_errors,
        fmt='none',
        ecolor='black',
        alpha=0.55,
        capsize=4,
        capthick=1.2,
        linewidth=1.2
    )
    
    # Set labels and title
    ax.set_ylabel('Enhancement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_title(
        'Enhancement Charge Rate by Race and Statute Level\n(with 95% Confidence Intervals)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45, ha='right', fontsize=10)
    
    # Add legend
    ax.legend(title='Statute Level', fontsize=10, title_fontsize=11, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add sample sizes and rates as text above bars
    max_y = 0
    for i in range(len(races)):
        # Misdemeanor
        if np.isfinite(mis_rates[i]):
            y_pos = mis_rates[i] + mis_errors[i] + 1
            ax.text(
                i - width/2, y_pos,
                f'{mis_rates[i]:.1f}%\n(n={int(mis_n[i]):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y = max(max_y, y_pos + 5)
        
        # Felony
        if np.isfinite(fel_rates[i]):
            y_pos = fel_rates[i] + fel_errors[i] + 1
            ax.text(
                i + width/2, y_pos,
                f'{fel_rates[i]:.1f}%\n(n={int(fel_n[i]):,})',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold'
            )
            max_y = max(max_y, y_pos + 5)
    
    # Set y-axis limits
    ax.set_ylim(0, min(100, max_y))
    
    fig.tight_layout()
    return fig


def plot_enhancement_rate_by_race_statute_category(enhancement_by_primary, top_categories=None):
    """
    Plot 3: Enhancement rates by race, statute level, and charge category (for selected categories).
    
    Parameters
    ----------
    enhancement_by_primary : pd.DataFrame
        Enhancement data with columns: race_std, primary_statute_level, primary_charge_category, Enhanced, N
    top_categories : list, optional
        List of charge categories to include (e.g., ['DUI', 'Assault/Violence', 'Weapons'])
        If None, includes all categories.
        
    Returns
    -------
    matplotlib.figure.Figure
        Multi-panel figure with one subplot per charge category
    """
    df = enhancement_by_primary.copy()
    
    # Filter to specific categories if provided
    if top_categories is not None:
        df = df[df['primary_charge_category'].isin(top_categories)]
        categories = top_categories
    else:
        categories = df['primary_charge_category'].unique().tolist()
    
    # Create subplots (one per category)
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(6*n_cats, 7))
    
    # Ensure axes is iterable
    if n_cats == 1:
        axes = [axes]
    
    fig.suptitle(
        'Enhancement Charge Rate by Race, Statute Level, and Charge Category\n(with 95% Confidence Intervals)',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    # Use centralized statute color map
    mis_color = STATUTE_COLOR_MAP['Misdemeanor']
    fel_color = STATUTE_COLOR_MAP['Felony']
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        
        # Filter to this category
        cat_data = df[df['primary_charge_category'] == category].copy()
        
        # Aggregate by race and statute level
        summary = (
            cat_data
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
        
        # Get unique races present in this category (preserving RACE_ORDER)
        mis_races = set(mis['race_std'].dropna().astype(str).tolist())
        fel_races = set(fel['race_std'].dropna().astype(str).tolist())
        all_races_set = mis_races.union(fel_races)
        
        # Filter RACE_ORDER to only include races present in this category
        all_races = [r for r in RACE_ORDER if r in all_races_set]
        
        # Prepare data arrays
        mis_dict = {r: {'rate': 0, 'se': 0, 'n': 0} for r in all_races}
        fel_dict = {r: {'rate': 0, 'se': 0, 'n': 0} for r in all_races}
        
        for _, row in mis.iterrows():
            r = str(row['race_std'])
            if r in all_races:
                mis_dict[r] = {
                    'rate': row['Enhancement Rate'] * 100,
                    'se': row['SE'] * 100,
                    'n': row['N']
                }
        
        for _, row in fel.iterrows():
            r = str(row['race_std'])
            if r in all_races:
                fel_dict[r] = {
                    'rate': row['Enhancement Rate'] * 100,
                    'se': row['SE'] * 100,
                    'n': row['N']
                }
        
        races = all_races
        mis_rates = np.array([mis_dict[r]['rate'] for r in races])
        fel_rates = np.array([fel_dict[r]['rate'] for r in races])
        mis_se = np.array([mis_dict[r]['se'] for r in races])
        fel_se = np.array([fel_dict[r]['se'] for r in races])
        mis_n = np.array([mis_dict[r]['n'] for r in races])
        fel_n = np.array([fel_dict[r]['n'] for r in races])
        
        # Calculate 95% CI
        mis_errors = mis_se * Z
        fel_errors = fel_se * Z
        
        # Create grouped bars
        x = np.arange(len(races))
        width = 0.35
        
        # Only plot bars where n > 0
        mis_mask = mis_n > 0
        fel_mask = fel_n > 0
        
        if mis_mask.any():
            ax.bar(
                x[mis_mask] - width/2, mis_rates[mis_mask],
                width,
                label='Misdemeanor',
                color=mis_color,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.2
            )
            
            ax.errorbar(
                x[mis_mask] - width/2, mis_rates[mis_mask],
                yerr=mis_errors[mis_mask],
                fmt='none',
                ecolor='black',
                alpha=0.55,
                capsize=3,
                capthick=1,
                linewidth=1
            )
        
        if fel_mask.any():
            ax.bar(
                x[fel_mask] + width/2, fel_rates[fel_mask],
                width,
                label='Felony',
                color=fel_color,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.2
            )
            
            ax.errorbar(
                x[fel_mask] + width/2, fel_rates[fel_mask],
                yerr=fel_errors[fel_mask],
                fmt='none',
                ecolor='black',
                alpha=0.55,
                capsize=3,
                capthick=1,
                linewidth=1
            )
        
        # Set labels and title
        ax.set_ylabel('Enhancement Rate (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Race', fontsize=11, fontweight='bold')
        ax.set_title(
            f'{category}',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        
        # Set x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=45, ha='right', fontsize=9)
        
        # Add legend (only on first subplot to avoid redundancy)
        if idx == 0:
            ax.legend(title='Statute Level', fontsize=9, title_fontsize=10, 
                     frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add sample sizes and rates as text above bars
        max_y = 0
        for i in range(len(races)):
            # Misdemeanor
            if mis_mask[i]:
                y_pos = mis_rates[i] + mis_errors[i] + 1
                ax.text(
                    i - width/2, y_pos,
                    f'{mis_rates[i]:.1f}%\n(n={int(mis_n[i]):,})',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    fontweight='bold'
                )
                max_y = max(max_y, y_pos + 8)
            
            # Felony
            if fel_mask[i]:
                y_pos = fel_rates[i] + fel_errors[i] + 1
                ax.text(
                    i + width/2, y_pos,
                    f'{fel_rates[i]:.1f}%\n(n={int(fel_n[i]):,})',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    fontweight='bold'
                )
                max_y = max(max_y, y_pos + 8)
        
        # Set y-axis limits
        if max_y > 0:
            ax.set_ylim(0, min(100, max_y))
        else:
            ax.set_ylim(0, 100)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig

# ------------------------------------------------------------------
# Prosecution: Wobbler Felony Rates
# ------------------------------------------------------------------

def plot_wobbler_felony_rates(wobbler_summary):
    """
    Bar chart showing felony filing rates for wobbler charges by race with 95% CI.
    """
    df = wobbler_summary.copy()
    
    # Ensure index is a column for easier processing
    if df.index.name == 'Canonical Race':
        df = df.reset_index()
    
    # Apply race ordering
    df['Canonical Race'] = pd.Categorical(df['Canonical Race'], categories=RACE_ORDER, ordered=True)
    df = df.sort_values('Canonical Race')
    
    # Extract data
    races = df['Canonical Race'].astype(str).to_list()
    felony_rates = df['Felony Rate'].to_numpy()
    se = df['Felony Rate SE'].to_numpy()
    totals = df['Total'].to_numpy()
    
    # Convert to percentage
    felony_rates = felony_rates * 100
    se = se * 100
    
    # Calculate 95% CI
    errors = se * Z
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(races))
    colors = [COLOR_MAP.get(r, "#7f7f7f") for r in races]
    
    # Create bars
    ax.bar(
        x, felony_rates,
        color=colors,
        alpha=0.85,
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add error bars
    ax.errorbar(
        x, felony_rates,
        yerr=errors,
        fmt='none',
        ecolor='black',
        alpha=0.55,
        capsize=5,
        capthick=1.5,
        linewidth=1.5
    )
    
    # Set labels and title
    ylabel = 'Wobbler Charged as Felony (%)'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Canonical Race', fontsize=12, fontweight='bold')
    ax.set_title(
        'Wobbler Charges: Felony Filing Rate by Race\n(with 95% Confidence Intervals)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add sample sizes as text above bars
    for i, (rate, err, total) in enumerate(zip(felony_rates, errors, totals)):
        if np.isfinite(rate):
            ax.text(
                i, rate + err + 2,
                f'n={int(total):,}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    # Set y-axis limits to prevent label clipping
    ymax = np.nanmax(felony_rates + errors)
    if np.isfinite(ymax):
        ax.set_ylim(0, min(100, ymax * 1.15))
    else:
        ax.set_ylim(0, 100)
    
    fig.tight_layout()
    return fig