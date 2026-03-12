# -----------------------------------------------------------------------------
# Statistcal calculation utilities for policing rates and sensitivity analyses
# -----------------------------------------------------------------------------

import numpy as np

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



def summarize_agency_black_white_hit_rates(df):
    """
    Agency-level White vs Black hit-rate comparison.
    Hit rate is calculated among searched cases only.
    
    Filters out agencies with fewer than 5 searches for either White or 
    Black/African American individuals to ensure more stable rate estimates.
    """

    tmp = df.copy()

    # Keep only searched cases
    tmp = tmp[tmp["disc_search"]].copy()

    # Keep only White and Black rows
    tmp = tmp[tmp["race_std"].isin(["White", "Black/African American"])].copy()

    # Summarize by agency x race
    summary = (
        tmp.groupby(["agency_name", "race_std"], as_index=False)
        .agg(
            Search_Count=("disc_hit", "size"),
            Hit_Count=("disc_hit", "sum"),
        )
    )

    summary["Hit_Rate"] = summary["Hit_Count"] / summary["Search_Count"]

    # Split into White and Black summaries, then merge
    white = (
        summary[summary["race_std"] == "White"]
        [["agency_name", "Search_Count", "Hit_Count", "Hit_Rate"]]
        .rename(columns={
            "Search_Count": "White_Search_Count",
            "Hit_Count": "White_Hit_Count",
            "Hit_Rate": "White_Hit_Rate",
        })
    )

    black = (
        summary[summary["race_std"] == "Black/African American"]
        [["agency_name", "Search_Count", "Hit_Count", "Hit_Rate"]]
        .rename(columns={
            "Search_Count": "Black_Search_Count",
            "Hit_Count": "Black_Hit_Count",
            "Hit_Rate": "Black_Hit_Rate",
        })
    )

    out = white.merge(black, on="agency_name", how="inner")

    # Filter out agencies with fewer than 5 searches for either group
    out = out[
        (out["White_Search_Count"] >= 5) & 
        (out["Black_Search_Count"] >= 5)
    ].copy()

    out["Avg_Search_Count"] = (
        out["White_Search_Count"] + out["Black_Search_Count"]
    ) / 2

    # Rename and properly capitalize agency names
    agency_name_map = {
        "CA ST UNIV PD-FULLERTON": "Cal State Fullerton PD",
        "UC-IRVINE PD": "UC Irvine PD",
        "ORANGE CO SO": "Orange County Sheriff's Office"
    }
    
    out["agency_name"] = out["agency_name"].replace(agency_name_map)
    
    # Capitalize agency names: title case for words, but keep PD and SO in caps
    def format_agency_name(name):
        # Skip if already manually mapped
        if name in agency_name_map.values():
            return name
        
        # Title case the name
        words = name.split()
        formatted_words = []
        for word in words:
            # Keep PD, SO, CO in all caps
            if word.upper() in ['PD', 'SO', 'CO']:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.title())
        
        return ' '.join(formatted_words)
    
    out["agency_name"] = out["agency_name"].apply(format_agency_name)

    return out



def policing_rates_by_reason_for_contact(df, census):
    """
    Calculate policing rates stratified by reason for contact.
    """
    reason_types = [
        "Moving violation",
        "Equipment violation",
        "Non-moving violation",
        "Suspect criminal activity"
    ]
    
    policing_by_reason = {}
    
    for reason in reason_types:
        subset = df[df["reason_for_contact"] == reason].copy()
        if len(subset) > 0:
            rates = policing_rates(subset, census)
            policing_by_reason[reason] = rates
    
    return policing_by_reason



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