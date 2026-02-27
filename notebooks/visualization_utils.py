import pandas as pd
import matplotlib.pyplot as plt
import os

def visualization_setup(df, race_col):
    """
    Enforce a consistent race order and sort for plotting.
    race_col should be either:
      - "Perceived Race" (policing)
      - "Canonical Race" (prosecution)
    """
    out = df.copy()

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]

    if race_col in out.columns:
        out[race_col] = pd.Categorical(out[race_col], categories=race_order, ordered=True)

    if "Year" in out.columns and race_col in out.columns:
        out = out.sort_values(["Year", race_col])

    return out


def visualize_policing(policing_analysis):
    df = policing_analysis.copy()
    race_col = "Perceived Race"
    
    # Import the visualization_setup function
    from visualization_utils import visualization_setup
    df = visualization_setup(df, race_col=race_col)
    
    # Define color palette (colorblind-friendly, Wong 2011)
    color_map = {
        "Black/African American": "#D55E00",  # vermillion
        "Hispanic/Latino": "#0072B2",  # blue
        "White": "#999999",  # gray (baseline)
        "Asian": "#009E73",  # bluish green
        "Other": "#CC79A7"  # reddish purple
    }
    
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
                
                # Add error bars if available and requested
                if show_errors:
                    se_col = f"{y_col} SE"
                    if se_col in d.columns and d[se_col].notna().any():
                        ax.errorbar(x_vals, d[y_col], yerr=d[se_col]*1.96,
                                  fmt='none', ecolor=color_map[race], 
                                  alpha=0.2, capsize=3, capthick=1.2)
                
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
                        for _, row in d.iterrows():
                            n = int(row[count_col])
                            x_pos = row["Year"]
                            ax.annotate(f'n={n:,}', 
                                      xy=(x_pos, row[y_col]),
                                      xytext=(0, 10), textcoords='offset points',
                                      fontsize=9, ha='center', alpha=0.8,
                                      color=color_map[race])
        
        ax.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(title="Perceived Race", fontsize=10, title_fontsize=11, 
                 frameon=True, fancybox=True, shadow=True, 
                 loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Format y-axis
        if "Rate" in y_col:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        
        # Add note about error bars
        if show_errors:
            fig.text(0.5, 0.02, 'Error bars show 95% confidence intervals', 
                    ha='center', fontsize=9, style='italic', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.05, 0.85, 1])  # Leave room for legend and footnote
        return fig
    
    # Create all four key figures
    figs = {}
    
    figs['stops'] = _create_figure(
        "Stops per 1,000",
        "Police Stop Rates by Perceived Race\n(per 1,000 residents)",
        "Stops per 1,000 Residents"
    )
    
    figs['searches'] = _create_figure(
        "Searches per 1,000",
        "Police Search Rates by Perceived Race\n(per 1,000 residents, 2020 Census)", 
        "Searches per 1,000 Residents"
    )
    
    figs['search_rate'] = _create_figure(
        "Search Rate",
        "Conditional Search Rate by Perceived Race\n(Among Those Stopped)",
        "Search Rate"
    )
    
    figs['hit_rate'] = _create_figure(
        "Hit Rate",
        "Contraband Hit Rate by Perceived Race\n(Outcome Test: Contraband Found Given Search)",
        "Hit Rate"
    )
    
    return figs


def visualize_prosecution(prosecution_analysis):
    df = prosecution_analysis.copy()
    race_col = "Canonical Race"
    
    # Import the visualization_setup function
    from visualization_utils import visualization_setup
    df = visualization_setup(df, race_col=race_col)
    
    # Define color palette (colorblind-friendly, Wong 2011)
    color_map = {
        "Black/African American": "#D55E00",  # vermillion
        "Hispanic/Latino": "#0072B2",  # blue
        "White": "#999999",  # gray (baseline)
        "Asian": "#009E73",  # bluish green
        "Other": "#CC79A7"  # reddish purple
    }
    
    years = sorted(df["Year"].dropna().unique().tolist())
    
    def _create_figure(dsub, y_col, title, ylabel, show_errors=True, show_n=True):
        """Create a single publication-quality line plot with error bars and sample sizes"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for race in color_map.keys():
            d = dsub[dsub[race_col] == race].sort_values("Year")
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
                
                # Add error bars if available and requested
                if show_errors:
                    se_col = f"{y_col} SE"
                    if se_col in d.columns and d[se_col].notna().any():
                        ax.errorbar(x_vals, d[y_col], yerr=d[se_col]*1.96,
                                  fmt='none', ecolor=color_map[race], 
                                  alpha=0.2, capsize=3, capthick=1.2)
                
                # Add sample size annotations if requested
                if show_n and "Total Charges" in d.columns:
                    for _, row in d.iterrows():
                        n = int(row["Total Charges"])
                        x_pos = row["Year"]
                        ax.annotate(f'n={n:,}', 
                                  xy=(x_pos, row[y_col]),
                                  xytext=(0, 10), textcoords='offset points',
                                  fontsize=9, ha='center', alpha=0.8,
                                  color=color_map[race])
        
        ax.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(title="Race", fontsize=10, title_fontsize=11, 
                 frameon=True, fancybox=True, shadow=True, 
                 loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout(rect=[0, 0.05, 0.85, 1])  # Leave room for legend and footnote
        return fig
    
    # Create figures for each statute level
    figs = {}
    
    for lvl in ["Felony", "Misdemeanor"]:
        dsub = df[df["statute_level"] == lvl].copy()
        if dsub.empty:
            continue
        
        figs[f'enhancement_{lvl.lower()}'] = _create_figure(
            dsub,
            "Enhancement Rate",
            f"Charge Enhancement Rate by Race\n({lvl} Charges)",
            "Enhancement Rate"
        )
    
    return figs

def create_sensitivity_visualization(strict_df, mixed_df):
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
        "Asian",
        "Other"
    ]

    # Sort both dataframes by race order
    strict = strict.set_index("Perceived Race").loc[race_order].reset_index()
    mixed = mixed.set_index("Perceived Race").loc[race_order].reset_index()

    # Create figure with improved styling
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'Sensitivity Analysis: Strict vs. Mixed Search Classification ({latest_year})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    x = range(len(strict))
    width = 0.4  # Increased from 0.35 to widen bars

    # Define colors
    strict_color = '#0072B2'  # Blue
    mixed_color = '#D55E00'   # Orange

    # ---- Panel A: Search Rates ----
    strict_search = strict['Search Rate'] * 100
    mixed_search = mixed['Search Rate'] * 100
    
    # Calculate standard errors in percentage points
    strict_se = strict['Search Rate SE'] * 100 * 1.96  # 95% CI
    mixed_se = mixed['Search Rate SE'] * 100 * 1.96

    bars1 = axes[0].bar(
        [i - width/2 for i in x],
        strict_search,
        width,
        label='Strict (discretionary only)',
        color=strict_color,
        alpha=0.8,
        yerr=strict_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
    )

    bars2 = axes[0].bar(
        [i + width/2 for i in x],
        mixed_search,
        width,
        label='Mixed (includes multi-basis)',
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

    bars3 = axes[1].bar(
        [i - width/2 for i in x],
        strict_hit,
        width,
        label='Strict (discretionary only)',
        color=strict_color,
        alpha=0.8,
        yerr=strict_hit_se,
        capsize=4,
        error_kw={'linewidth': 1.5, 'alpha': 0.6}
    )

    bars4 = axes[1].bar(
        [i + width/2 for i in x],
        mixed_hit,
        width,
        label='Mixed (includes multi-basis)',
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