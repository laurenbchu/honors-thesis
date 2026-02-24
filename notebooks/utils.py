import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_census(county_fips):
    """
    Load and relabel Decennial Census PL Table P2 data for a given county.

    Parameters
    ----------
    county_fips : str
        5-digit county FIPS code (e.g. "06059" for Orange County)

    Returns
    -------
    census_relabeled : pandas.DataFrame
        Census P2 table with readable column labels and NA columns removed
    """

    # Import census data from API
    data_url = (
        f"https://api.census.gov/data/2020/dec/pl"
        f"?get=group(P2)&ucgid=0500000US{county_fips}"
    )

    # Reformat to make data readable
    census = pd.read_json(data_url)
    census.columns = census.iloc[0]
    census = census.iloc[1:]

    # Import metadata for variable labels
    metadata_url = "https://api.census.gov/data/2020/dec/pl/variables.json"
    metadata = pd.read_json(metadata_url, typ="series")

    # Relabel each column with readable label of race/ethnicity group
    variables = metadata["variables"]
    meta = pd.DataFrame.from_dict(variables, orient="index")
    code_to_label = meta["label"].to_dict()

    # Remove columns with no data
    census = census.loc[:, ~census.columns.str.endswith("NA")]
    census_relabeled = census.rename(columns=code_to_label)

    return census_relabeled


def add_standardized_race(df, race_col):
    """
    Add race_std column that standardizes race labels across
    policing and prosecution datasets.
    """

    out = df.copy()

    mapping = {
        "Hispanic": "Hispanic/Latino",
        "Latinx": "Hispanic/Latino",
        "Black": "Black/African American",
        "Asian": "Asian",
        "White": "White",
    }

    out["race_std"] = (
        out[race_col]
        .map(mapping)
        .fillna("Other")
    )

    return out


def census_rollup(census_relabeled):
    """
    Convert a relabeled Decennial PL P2 census table to your coarse race population Series.

    This function:
      1) coerces all non-id columns to numeric
      2) returns a coarse population Series with keys:
         ["Hispanic/Latino", "White", "Black/African American", "Asian", "Other"]

    Parameters
    ----------
    census_relabeled : pandas.DataFrame
        Census P2 table with readable labels (after renaming columns)

    Returns
    -------
    pandas.Series
        Coarse race populations
    """

    id_cols = ["NAME", "Geography", "Uniform Census Geography Identifier clause"]

    df = census_relabeled.copy()

    # Coerce the non-id columns to numeric types for easier computation
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pop = df.iloc[0]

    # Relabel/add together the census to be 5 categories
    return pd.Series({
        "Hispanic/Latino": pop[" !!Total:!!Hispanic or Latino"],
        "White": pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone"],
        "Black/African American": pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!Black or African American alone"],
        "Asian": pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!Asian alone"],
        "Other": (
            pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!American Indian and Alaska Native alone"]
            + pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!Native Hawaiian and Other Pacific Islander alone"]
            + pop[" !!Total:!!Not Hispanic or Latino:!!Population of one race:!!Some Other Race alone"]
            + pop[" !!Total:!!Not Hispanic or Latino:!!Population of two or more races:"]
        ),
    })


def policing_rates(df, census):
    """
    Build a race × year policing summary table for discretionary stops only,
    measuring discretionary search behavior.
    """

    g = df.groupby(["year", "race_std"])

    summary = g.agg(
        Stop_Count=("disc_search", "size"),
        Search_Count=("disc_search", "sum"),
        Hit_Count=("disc_hit", "sum"),
    ).reset_index()

    # Rename columns
    summary = summary.rename(columns={
        "year": "Year",
        "race_std": "Perceived Race",
        "Stop_Count": "Stop Count",
        "Search_Count": "Search Count",
        "Hit_Count": "Hit Count",
    })

    # 5. Rates
    summary["Search Rate"] = (
        summary["Search Count"] / summary["Stop Count"]
    )

    summary["Hit Rate"] = np.where(
        summary["Search Count"] > 0,
        summary["Hit Count"] / summary["Search Count"],
        np.nan
    )

    # 6. Add population
    summary["Population"] = summary["Perceived Race"].map(census)

    # Per-capita rates (per 1,000 residents)
    per = 1000
    summary["Stops per 1,000"] = (
        summary["Stop Count"] / summary["Population"]
    ) * per

    summary["Searches per 1,000"] = (
        summary["Search Count"] / summary["Population"]
    ) * per

    return summary


def prosecution_rates_by_statute_level(prosecution):
    """
    Build a race × year × statute_level prosecution summary table.

    Returns:
        - Total Charges
        - Charge Rate
        - Enhancement Rate
    """

    df = prosecution.copy()

    # Ensure standardized race column exists
    if "Canonical Race" not in df.columns and "race_std" in df.columns:
        df = df.rename(columns={"race_std": "Canonical Race"})

    # Remove Infractions
    df = df[df["statute_level"] != "Infraction"]

    summary = (
        df.groupby(["Canonical Race", "Year", "statute_level"])
          .agg(
              **{
                  "Total Charges": ("was_filed_by_da", "size"),
                  "Enhancement Rate": ("is_enhancement_charge", "mean"),
              }
          )
          .reset_index()
    )

    return summary


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

def format_policing_table_for_thesis(policing_analysis):
    df = policing_analysis.copy()
    
    # Reorder columns for logical flow
    df = df[[
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
    
    # Rename columns for thesis
    df = df.rename(columns={
        'Perceived Race': 'Race/Ethnicity',
        'Stop Count': 'Total Stops',
        'Search Count': 'Total Searches',
        'Hit Count': 'Contraband Found',
        'Stops per 1,000': 'Stop Rate\n(per 1,000)',
        'Searches per 1,000': 'Search Rate\n(per 1,000)',
        'Search Rate': 'Conditional\nSearch Rate',
        'Hit Rate': 'Hit Rate'
    })
    
    # Format numbers appropriately
    df['Population'] = df['Population'].apply(lambda x: f'{int(x):,}')
    df['Total Stops'] = df['Total Stops'].apply(lambda x: f'{int(x):,}')
    df['Total Searches'] = df['Total Searches'].apply(lambda x: f'{int(x):,}')
    df['Contraband Found'] = df['Contraband Found'].apply(lambda x: f'{int(x):,}')
    
    # Format rates as percentages or rounded decimals
    df['Stop Rate\n(per 1,000)'] = df['Stop Rate\n(per 1,000)'].apply(lambda x: f'{x:.1f}')
    df['Search Rate\n(per 1,000)'] = df['Search Rate\n(per 1,000)'].apply(lambda x: f'{x:.1f}')
    df['Conditional\nSearch Rate'] = df['Conditional\nSearch Rate'].apply(lambda x: f'{x:.1%}')
    df['Hit Rate'] = df['Hit Rate'].apply(lambda x: f'{x:.1%}' if pd.notna(x) else '—')
    
    return df


def format_prosecution_table_for_thesis(prosecution_analysis):
    df = prosecution_analysis.copy()
    
    # Rename columns for thesis
    df = df.rename(columns={
        'Canonical Race': 'Race/Ethnicity',
        'Year': 'Year',
        'statute_level': 'Charge Level',
        'Total Charges': 'Total Charges',
        'Enhancement Rate': 'Enhancement Rate'
    })
    
    # Reorder columns
    df = df[['Year', 'Charge Level', 'Race/Ethnicity', 'Total Charges', 'Enhancement Rate']]
    
    # Format numbers
    df['Total Charges'] = df['Total Charges'].apply(lambda x: f'{int(x):,}')
    df['Enhancement Rate'] = df['Enhancement Rate'].apply(lambda x: f'{x:.1%}')
    
    return df


def create_disparity_ratio_table(policing_analysis, baseline_race='White'):
    """
    Create a table showing disparity ratios relative to White baseline
    This is very useful for thesis to highlight key findings
    """
    df = policing_analysis.copy()
    
    # Get the most recent year
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    # Get baseline values
    baseline = df_latest[df_latest['Perceived Race'] == baseline_race].iloc[0]
    
    # Calculate ratios for each race
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
    
    # Format ratios
    for col in ['Stop Rate Ratio', 'Search Rate Ratio', 'Conditional Search Ratio', 'Hit Rate Ratio']:
        ratio_df[col] = ratio_df[col].apply(lambda x: f'{x:.2f}×' if pd.notna(x) else '—')
    
    return ratio_df


def save_tables_for_thesis(policing_analysis, prosecution_analysis, output_dir='../output'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Format tables
    policing_formatted = format_policing_table_for_thesis(policing_analysis)
    prosecution_formatted = format_prosecution_table_for_thesis(prosecution_analysis)
    disparity_ratios = create_disparity_ratio_table(policing_analysis)
    
    # Save as CSV (for reference)
    policing_formatted.to_csv(f'{output_dir}/policing_table_formatted.csv', index=False)
    prosecution_formatted.to_csv(f'{output_dir}/prosecution_table_formatted.csv', index=False)
    disparity_ratios.to_csv(f'{output_dir}/disparity_ratios.csv', index=False)
    
    # Save as LaTeX (for direct thesis inclusion)
    with open(f'{output_dir}/policing_table.tex', 'w') as f:
        f.write(policing_formatted.to_latex(index=False, escape=False, 
                                            caption='Police Stop and Search Rates by Race/Ethnicity (2022-2024)',
                                            label='tab:policing'))
    
    with open(f'{output_dir}/prosecution_table.tex', 'w') as f:
        f.write(prosecution_formatted.to_latex(index=False, escape=False,
                                               caption='Charge Enhancement Rates by Race/Ethnicity (2021-2023)',
                                               label='tab:prosecution'))
    
    with open(f'{output_dir}/disparity_ratios.tex', 'w') as f:
        f.write(disparity_ratios.to_latex(index=False, escape=False,
                                          caption='Disparity Ratios Relative to White Baseline (2024)',
                                          label='tab:disparities'))
    
    # Save as Excel (for easy viewing/editing)
    with pd.ExcelWriter(f'{output_dir}/thesis_tables.xlsx') as writer:
        policing_formatted.to_excel(writer, sheet_name='Policing', index=False)
        prosecution_formatted.to_excel(writer, sheet_name='Prosecution', index=False)
        disparity_ratios.to_excel(writer, sheet_name='Disparity Ratios', index=False)
    
    return policing_formatted, prosecution_formatted, disparity_ratios
