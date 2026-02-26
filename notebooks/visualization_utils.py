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
    from utils import visualization_setup
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
    
    def _create_figure(y_col, title, ylabel, caption):
        """Create a single publication-quality line plot with caption"""
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
                 frameon=True, fancybox=True, shadow=True, loc='best')
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Format y-axis
        if "Rate" in y_col:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        
        # Add caption below the plot
        fig.text(0.5, 0.01, caption, ha='center', fontsize=9, 
                style='italic', wrap=True, color='#444444')
        
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        return fig
    
    # Create all four key figures with descriptive captions
    figs = {}
    
    figs['stops'] = _create_figure(
        "Stops per 1,000",
        "Police Stop Rates by Perceived Race\n(per 1,000 residents)",
        "Stops per 1,000 Residents",
        caption=("Population-normalized exposure to discretionary police stops. "
                "Rates based on 2020 Census population data. "
                "White baseline shown as dashed line. ")
    )
    
    figs['searches'] = _create_figure(
        "Searches per 1,000",
        "Police Search Rates by Perceived Race\n(per 1,000 residents, 2020 Census)", 
        "Searches per 1,000 Residents",
        caption=("Population-normalized exposure to purely discretionary police searches. "
                "White baseline shown as dashed line.")
    )
    
    figs['search_rate'] = _create_figure(
        "Search Rate",
        "Conditional Search Rate by Perceived Race\n(among those stopped)",
        "Search Rate",
        caption=("Conditional probability of being searched for a purely discretionary reason, given a discretionary stop. "
                " White baseline shown as dashed line.")
    )
    
    figs['hit_rate'] = _create_figure(
        "Hit Rate",
        "Contraband Hit Rate by Perceived Race\n(outcome test: contraband found given search)",
        "Hit Rate",
        caption=("Outcome test: percentage of purely discretionary searches that yield contraband. "
                "White baseline shown as dashed line.")
    )
    
    return figs


def visualize_prosecution(prosecution_analysis):
    df = prosecution_analysis.copy()
    race_col = "Canonical Race"
    
    # Import the visualization_setup function
    from utils import visualization_setup
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
    
    def _create_figure(dsub, y_col, title, ylabel, caption):
        """Create a single publication-quality line plot with caption"""
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
                 frameon=True, fancybox=True, shadow=True, loc='best')
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add caption below the plot
        fig.text(0.5, 0.01, caption, ha='center', fontsize=9, 
                style='italic', wrap=True, color='#444444')
        
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        return fig
    
    # Create figures for each statute level with descriptive captions
    figs = {}
    
    for lvl in ["Felony", "Misdemeanor"]:
        dsub = df[df["statute_level"] == lvl].copy()
        if dsub.empty:
            continue
        
        if lvl == "Felony":
            caption = ("Enhancement rate for felony charges by defendant race. Enhancement charges add additional "
                      "time to a person's sentence. "
                      "White baseline shown as dashed line. Colors follow Wong (2011) palette.")
        else:  # Misdemeanor
            caption = ("Enhancement rate for misdemeanor charges by defendant race. Enhancement charges add additional "
                      "time to a person's sentence. "
                      "White baseline shown as dashed line.")
        
        figs[f'enhancement_{lvl.lower()}'] = _create_figure(
            dsub,
            "Enhancement Rate",
            f"Charge Enhancement Rate by Race\n({lvl} charges)",
            "Enhancement Rate",
            caption=caption
        )
    
    return figs


def export_tables_to_latex(policing_analysis, prosecution_analysis, output_dir='../output'):
    """
    Export formatted tables directly as LaTeX files for thesis inclusion.
    Generates clean, publication-ready LaTeX tables with optimal formatting.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================================
    # POLICING TABLE
    # ============================================================================
    policing_df = policing_analysis.copy()
    policing_df = policing_df[[
        'Year', 
        'Perceived Race', 
        'Population',
        'Stop Count', 
        'Stops per 1,000',
        'Search Count',
        'Searches per 1,000',
        'Search Rate',
        'Hit Count',
        'Hit Rate'
    ]]
    
    policing_df = policing_df.rename(columns={
        'Perceived Race': 'Race/Ethnicity',
        'Stop Count': 'Total Stops',
        'Search Count': 'Total Searches',
        'Hit Count': 'Contraband Found',
        'Stops per 1,000': 'Stop Rate (per 1,000)',
        'Searches per 1,000': 'Search Rate (per 1,000)',
        'Search Rate': 'Conditional Search Rate',
        'Hit Rate': 'Hit Rate'
    })
    
    # Format numbers for LaTeX
    policing_df['Population'] = policing_df['Population'].apply(lambda x: f'{int(x):,}')
    policing_df['Total Stops'] = policing_df['Total Stops'].apply(lambda x: f'{int(x):,}')
    policing_df['Total Searches'] = policing_df['Total Searches'].apply(lambda x: f'{int(x):,}')
    policing_df['Contraband Found'] = policing_df['Contraband Found'].apply(lambda x: f'{int(x):,}')
    policing_df['Stop Rate (per 1,000)'] = policing_df['Stop Rate (per 1,000)'].apply(lambda x: f'{x:.1f}')
    policing_df['Search Rate (per 1,000)'] = policing_df['Search Rate (per 1,000)'].apply(lambda x: f'{x:.1f}')
    policing_df['Conditional Search Rate'] = policing_df['Conditional Search Rate'].apply(lambda x: f'{x*100:.1f}\\%')
    policing_df['Hit Rate'] = policing_df['Hit Rate'].apply(lambda x: f'{x*100:.1f}\\%' if pd.notna(x) else '---')
    
    latex_str = policing_df.to_latex(
        index=False, 
        escape=False,
        caption='Police Stop and Search Rates by Race/Ethnicity (2022--2024)',
        label='tab:policing',
        column_format='ccrrrrrrr',
        position='htbp'
    )
    
    # Improve LaTeX formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\\bottomrule')
    
    with open(f'{output_dir}/policing_table.tex', 'w') as f:
        f.write(latex_str)
    
    # ============================================================================
    # PROSECUTION TABLE
    # ============================================================================
    prosecution_df = prosecution_analysis.copy()
    prosecution_df = prosecution_df.rename(columns={
        'Canonical Race': 'Race/Ethnicity',
        'statute_level': 'Charge Level',
        'Enhancement Rate': 'Enhancement Rate'
    })
    prosecution_df = prosecution_df[['Year', 'Charge Level', 'Race/Ethnicity', 'Total Charges', 'Enhancement Rate']]
    
    prosecution_df['Total Charges'] = prosecution_df['Total Charges'].apply(lambda x: f'{int(x):,}')
    prosecution_df['Enhancement Rate'] = prosecution_df['Enhancement Rate'].apply(lambda x: f'{x*100:.1f}\\%')
    
    latex_str = prosecution_df.to_latex(
        index=False, 
        escape=False,
        caption='Charge Enhancement Rates by Race/Ethnicity (2021--2023)',
        label='tab:prosecution',
        column_format='cclrr',
        position='htbp'
    )
    
    # Improve LaTeX formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\\bottomrule')
    
    with open(f'{output_dir}/prosecution_table.tex', 'w') as f:
        f.write(latex_str)
    
    # ============================================================================
    # DISPARITY RATIOS TABLE
    # ============================================================================
    disparity_df = policing_analysis.copy()
    latest_year = int(disparity_df['Year'].max())
    df_latest = disparity_df[disparity_df['Year'] == latest_year].copy()
    baseline = df_latest[df_latest['Perceived Race'] == 'White'].iloc[0]
    
    ratios = []
    for _, row in df_latest.iterrows():
        race = row['Perceived Race']
        ratios.append({
            'Race/Ethnicity': race,
            'Stop Rate Ratio': row['Stops per 1,000'] / baseline['Stops per 1,000'],
            'Search Rate Ratio': row['Searches per 1,000'] / baseline['Searches per 1,000'],
            'Conditional Search Ratio': row['Search Rate'] / baseline['Search Rate'],
            'Hit Rate Ratio': row['Hit Rate'] / baseline['Hit Rate'] if pd.notna(row['Hit Rate']) else None
        })
    
    ratio_df = pd.DataFrame(ratios)
    
    for col in ['Stop Rate Ratio', 'Search Rate Ratio', 'Conditional Search Ratio', 'Hit Rate Ratio']:
        ratio_df[col] = ratio_df[col].apply(lambda x: f'{x:.2f}$\\times$' if pd.notna(x) else '---')
    
    latex_str = ratio_df.to_latex(
        index=False, 
        escape=False,
        caption=f'Disparity Ratios Relative to White Baseline ({latest_year})',
        label='tab:disparities',
        column_format='lcccc',
        position='htbp'
    )
    
    # Improve LaTeX formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\\bottomrule')
    
    with open(f'{output_dir}/disparity_ratios.tex', 'w') as f:
        f.write(latex_str)