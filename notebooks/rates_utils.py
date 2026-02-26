import numpy as np
import pandas as pd

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


def policing_rates_mixed(df, census):
    """
    Build a race × year policing summary table for discretionary stops,
    treating MIXED search bases as discretionary (sensitivity analysis).
    
    This alternative classification includes searches with both discretionary
    and nondiscretionary bases in the discretionary search count.
    """

    g = df.groupby(["year", "race_std"])

    summary = g.agg(
        Stop_Count=("disc_search_mixed", "size"),
        Search_Count=("disc_search_mixed", "sum"),
        Hit_Count=("disc_hit_mixed", "sum"),
    ).reset_index()

    # Rename columns
    summary = summary.rename(columns={
        "year": "Year",
        "race_std": "Perceived Race",
        "Stop_Count": "Stop Count",
        "Search_Count": "Search Count",
        "Hit_Count": "Hit Count",
    })

    # Rates
    summary["Search Rate"] = (
        summary["Search Count"] / summary["Stop Count"]
    )

    summary["Hit Rate"] = np.where(
        summary["Search Count"] > 0,
        summary["Hit Count"] / summary["Search Count"],
        np.nan
    )

    # Add population
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


def compare_search_classifications(policing_strict, policing_mixed):
    """
    Create comparison tables showing differences between strict and mixed
    search classification approaches.
    
    Parameters
    ----------
    policing_strict : DataFrame
        Results using strict classification (discretionary only)
    policing_mixed : DataFrame
        Results using mixed classification (discretionary + mixed)
    
    Returns
    -------
    dict
        Dictionary containing comparison DataFrames:
        - 'by_year_race': Full comparison by year and race
        - 'summary_2024': Summary for most recent year (2024)
        - 'disparity_comparison': Disparity ratios under both approaches
    """
    
    # Merge the two approaches
    comparison = policing_strict.merge(
        policing_mixed,
        on=['Year', 'Perceived Race', 'Population'],
        suffixes=('_strict', '_mixed')
    )
    
    # Calculate differences
    comparison['Search Count Diff'] = (
        comparison['Search Count_mixed'] - comparison['Search Count_strict']
    )
    comparison['Search Count % Change'] = (
        comparison['Search Count Diff'] / comparison['Search Count_strict'] * 100
    )
    
    comparison['Search Rate Diff'] = (
        comparison['Search Rate_mixed'] - comparison['Search Rate_strict']
    )
    comparison['Search Rate % Change'] = (
        comparison['Search Rate Diff'] / comparison['Search Rate_strict'] * 100
    )
    
    comparison['Hit Rate Diff'] = (
        comparison['Hit Rate_mixed'] - comparison['Hit Rate_strict']
    )
    
    # Create summary for most recent year
    latest_year = comparison['Year'].max()
    summary_2024 = comparison[comparison['Year'] == latest_year].copy()
    
    # Calculate disparity ratios for both approaches
    def calc_disparities(df, suffix):
        baseline = df[df['Perceived Race'] == 'White'].iloc[0]
        ratios = []
        for _, row in df.iterrows():
            ratios.append({
                'Perceived Race': row['Perceived Race'],
                f'Search Rate Ratio{suffix}': row[f'Search Rate{suffix}'] / baseline[f'Search Rate{suffix}'],
                f'Hit Rate Ratio{suffix}': row[f'Hit Rate{suffix}'] / baseline[f'Hit Rate{suffix}']
                    if pd.notna(row[f'Hit Rate{suffix}']) and pd.notna(baseline[f'Hit Rate{suffix}']) else np.nan
            })
        return pd.DataFrame(ratios)
    
    strict_ratios = calc_disparities(summary_2024, '_strict')
    mixed_ratios = calc_disparities(summary_2024, '_mixed')
    
    disparity_comparison = strict_ratios.merge(
        mixed_ratios[['Perceived Race', 'Search Rate Ratio_mixed', 'Hit Rate Ratio_mixed']],
        on='Perceived Race'
    )
    
    return {
        'by_year_race': comparison,
        'summary_2024': summary_2024,
        'disparity_comparison': disparity_comparison
    }
