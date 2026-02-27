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
    
    Includes standard errors for rates and per-capita metrics.
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

    # Calculate rates
    summary["Search Rate"] = (
        summary["Search Count"] / summary["Stop Count"]
    )

    summary["Hit Rate"] = np.where(
        summary["Search Count"] > 0,
        summary["Hit Count"] / summary["Search Count"],
        np.nan
    )

    # Calculate standard errors for rates using binomial proportion formula: SE = sqrt(p(1-p)/n)
    summary["Search Rate SE"] = np.sqrt(
        summary["Search Rate"] * (1 - summary["Search Rate"]) / summary["Stop Count"]
    )
    
    summary["Hit Rate SE"] = np.where(
        summary["Search Count"] > 0,
        np.sqrt(summary["Hit Rate"] * (1 - summary["Hit Rate"]) / summary["Search Count"]),
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
    
    # Standard errors for per-capita rates
    # Using Poisson approximation for count data: SE = sqrt(count) / population * 1000
    summary["Stops per 1,000 SE"] = (
        np.sqrt(summary["Stop Count"]) / summary["Population"]
    ) * per
    
    summary["Searches per 1,000 SE"] = (
        np.sqrt(summary["Search Count"]) / summary["Population"]
    ) * per

    return summary


def prosecution_rates_by_statute_level(prosecution):
    """
    Build a race × year × statute_level prosecution summary table.

    Returns:
        - Total Charges
        - Enhancement Rate
        - Standard Error
        - 95% Confidence Intervals
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
    
    # Calculate standard error: SE = sqrt(p * (1-p) / n)
    summary["Enhancement Rate SE"] = np.sqrt(
        (summary["Enhancement Rate"] * (1 - summary["Enhancement Rate"])) / 
        summary["Total Charges"]
    )
    
    # Calculate 95% confidence interval (±1.96 * SE)
    summary["Enhancement Rate CI Lower"] = (
        summary["Enhancement Rate"] - 1.96 * summary["Enhancement Rate SE"]
    ).clip(lower=0)
    
    summary["Enhancement Rate CI Upper"] = (
        summary["Enhancement Rate"] + 1.96 * summary["Enhancement Rate SE"]
    ).clip(upper=1)

    return summary


def policing_rates_mixed(df, census):
    """
    Build a race × year policing summary table for discretionary stops,
    treating mixed search bases as discretionary (sensitivity analysis).
    
    Includes standard errors for rates and per-capita metrics.
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

    # Calculate standard errors for rates
    summary["Search Rate SE"] = np.sqrt(
        summary["Search Rate"] * (1 - summary["Search Rate"]) / summary["Stop Count"]
    )
    
    summary["Hit Rate SE"] = np.where(
        summary["Search Count"] > 0,
        np.sqrt(summary["Hit Rate"] * (1 - summary["Hit Rate"]) / summary["Search Count"]),
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
    
    # Standard errors for per-capita rates
    summary["Stops per 1,000 SE"] = (
        np.sqrt(summary["Stop Count"]) / summary["Population"]
    ) * per
    
    summary["Searches per 1,000 SE"] = (
        np.sqrt(summary["Search Count"]) / summary["Population"]
    ) * per

    return summary


def calculate_enhancement_rates_by_category(df):
    """
    Calculate enhancement rates by race for each charge category.
    
    Includes standard errors and 95% confidence intervals.
    """
    
    # Group by race and charge category
    grouped = df.groupby(['race_std', 'charge_category']).agg({
        'is_enhancement_charge': ['sum', 'count', 'mean']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['Race', 'Charge Category', 'Enhancement Count', 'Total Count', 'Enhancement Rate']
    
    # Calculate standard error: SE = sqrt(p * (1-p) / n)
    grouped['Standard Error'] = np.sqrt(
        (grouped['Enhancement Rate'] * (1 - grouped['Enhancement Rate'])) / grouped['Total Count']
    )
    
    # Calculate 95% confidence interval (±1.96 * SE)
    grouped['CI Lower'] = grouped['Enhancement Rate'] - 1.96 * grouped['Standard Error']
    grouped['CI Upper'] = grouped['Enhancement Rate'] + 1.96 * grouped['Standard Error']
    
    # Ensure CI bounds are valid
    grouped['CI Lower'] = grouped['CI Lower'].clip(lower=0)
    grouped['CI Upper'] = grouped['CI Upper'].clip(upper=1)
    
    return grouped
