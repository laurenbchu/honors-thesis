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
    
    def _create_figure(y_col, title, ylabel):
        """Create a single publication-quality line plot"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for race in color_map.keys():
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                ax.plot(d["Year"], d[y_col], 
                       marker="o", markersize=8,
                       linewidth=linewidth, linestyle=linestyle,
                       color=color_map[race], label=race)
        
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
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave room for legend on right
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
    
    def _create_figure(dsub, y_col, title, ylabel):
        """Create a single publication-quality line plot"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for race in color_map.keys():
            d = dsub[dsub[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                linewidth = 2.5 if race == "White" else 2
                linestyle = '--' if race == "White" else '-'
                ax.plot(d["Year"], d[y_col], 
                       marker="o", markersize=8,
                       linewidth=linewidth, linestyle=linestyle,
                       color=color_map[race], label=race)
        
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
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave room for legend on right
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