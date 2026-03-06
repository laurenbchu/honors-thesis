# -----------------------------------------------------------------------------
# Statistcal calculation utilities for policing rates and sensitivity analyses
# -----------------------------------------------------------------------------

import numpy as np

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

    # Number of discretionary searches out of discretionary stops
    summary["Search Rate"] = (
        summary["Search Count"] / summary["Stop Count"]
    )

    # Number of hits out of discretionary searches
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



def policing_rates_sensitivity(df, census, sensitivity_type):
    """
    Build a race × year policing summary table for sensitivity analysis.
    
    Includes standard errors for rates and per-capita metrics.
    """

    search_col = f"disc_search_{sensitivity_type}"
    hit_col = f"disc_hit_{sensitivity_type}"

    g = df.groupby(["year", "race_std"])

    summary = g.agg(
        Stop_Count=(search_col, "size"),
        Search_Count=(search_col, "sum"),
        Hit_Count=(hit_col, "sum"),
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



def enhancement_rates_by_primary_severity(df):
    """
    Returns enhancement rates grouped by:
        primary_charge_category × race_std × Year × primary_statute_level
    """

    df = df.copy()

    from data_utils import assign_primary_charge_by_severity

    # Assign primary charge per case
    primary = assign_primary_charge_by_severity(df)

    # Collapse to one row per case
    case_level = (
        df.groupby("source_case_id", as_index=False)
          .agg(
              race_std=("race_std", "first"),
              Year=("Year", "first"),
              any_enhancement=("any_enhancement_in_case", "max")
          )
    )

    # Adds each case's primary charge category and statute level to the case-level table
    case_level = case_level.merge(primary, on="source_case_id", how="left")

    # Compute enhancement rates by group
    g = (
        case_level.groupby(
            ["primary_charge_category", "race_std", "Year", "primary_statute_level"],
            as_index=False
        )
        .agg(
            Enhanced=("any_enhancement", "sum"),
            N=("any_enhancement", "size")
        )
    )

    g["Enhancement Rate"] = g["Enhanced"] / g["N"]

    # Compute standard error and CI
    g["SE"] = np.sqrt(g["Enhancement Rate"] * (1 - g["Enhancement Rate"]) / g["N"])
    g["CI Lower"] = np.maximum(0, g["Enhancement Rate"] - 1.96 * g["SE"])
    g["CI Upper"] = np.minimum(1, g["Enhancement Rate"] + 1.96 * g["SE"])

    return g



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



def get_filtered_wobbler_categories(df):
    """
    Return charge categories to keep for wobbler analyses.

    Keeps categories that:
    - are wobblers,
    - are not 'Other',
    - have at least 500 total wobblers across all races,
    - have an overall felony filing rate of at least 5%.
    """
    wobblers = df[df["is_wobbler"]].copy()

    category_stats = (
        wobblers.groupby(["charge_category", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["Felony", "Misdemeanor"]:
        if col not in category_stats.columns:
            category_stats[col] = 0

    category_stats["Total"] = category_stats["Felony"] + category_stats["Misdemeanor"]
    category_stats["Felony Rate"] = category_stats["Felony"] / category_stats["Total"]

    keep_categories = (
        category_stats.loc[
            (category_stats["charge_category"] != "Other") &
            (category_stats["Total"] >= 500) &
            (category_stats["Felony Rate"] >= 0.05),
            ["charge_category", "Total"]
        ]
        .sort_values("Total", ascending=False)["charge_category"]
        .tolist()
    )

    return keep_categories