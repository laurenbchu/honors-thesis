# -------------------------------------------
# Data loading and preprocessing utilities
# -------------------------------------------

import pandas as pd

def load_census(county_fips):
    """
    Load and relabel Decennial Census PL Table P2 data for a given county.
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
    Convert a relabeled Decennial PL P2 census table to the coarse race population Series.
    "Hispanic/Latino", "White", "Black/African American", "Asian", "Other"
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



def categorize_charge(charge_desc):
    """
    Categorize SUBSTANTIVE charges into broad crime types.
    Note: This should only be called on non-enhancement charges.
    """
    if pd.isna(charge_desc):
        return 'Other'
    
    charge_desc = str(charge_desc).upper()

    # Drug possession
    if any(word in charge_desc for word in ['POSSESS CONTROLLED', 'POSSESS NARCOTIC', 
        'POSSESS UNLAWFUL PARAPHERNALIA', 'USE/UNDER INFLUENCE OF CONTROLLED SUBSTANCE', 
        'BRING CONTROLLED SUBSTANCE', 'NITROUS OXIDE', 'TOLUENE', 'POSS DRGS', 'BRING DRGS', 
        'BRING DRUGS', 'NARCOTIC', 'CONTROLLED', 'MARIJUANA', 'COCAINE', 'FALSE COMPARTMENT',
        'TRANSPORT/ETC CONTROLLED SUBSTANCE', 'TRANSPORT/SELL NARCOTIC/CONTROLLED SUBSTANCE']):
        return 'Drug Possession/Paraphernalia'
    
    # DUI related
    if any(word in charge_desc for word in ['DUI', 'DRIVING UNDER INFLUENCE', '0.08', '0.05', 'WHILE INTOXICATED']):
        return 'DUI'

    # Resist/obstruct
    elif any(word in charge_desc for word in ['OBSTRUCT', 'RESIST', 'GIVE FALSE ID TO PO', 
        'DESTROY/CONCEAL', 'EVADING PEACE OFFICER', 'FALSE IDENTIFICATION', 'FALSE INFORMATION', 'DISSUADE',
        'FAIL TO PRVD', 'FAIL TO PROVIDE']):
        return 'Obstruct/Resist Officer'

    # Theft
    elif any(word in charge_desc for word in ['THEFT', 'BURGLARY', 'ROBBERY', 'STOLEN', 
        'SHOPLIFTING', 'APPROPRIATE LOST PROPERTY', 'BURG', 'CARJACKING', 'EMBEZZLEMENT', 'UNAUTH ENTR', 
        'UNAUTH ENTRY', 'THFT']):
        return 'Theft/Burglary/Robbery'

    # Vandalism
    elif any(word in charge_desc for word in ['VANDALISM', 'ARSON', 'DAMAGE/DESTROY', 'VANDALIZE', 'CAUSING FIRE']):
        return 'Vandalism'

    # Vehicle related (non-DUI)
    elif any(word in charge_desc for word in ['VEHICLE', 'DRIVING WITHOUT', 'FOLLOWING ACCIDENT', 
        'RECKLESS DRIVING', 'EVADING A PEACE OFFICER:WRONG WAY DRIVER', 'HIT AND RUN', 'SPEED CONTEST', 
        'DISABLED PERSON PLACARD']):
        return 'Vehicle-Related (non-DUI)'

    # Violence
    elif any(word in charge_desc for word in ['ASSAULT', 'BATTERY', 'VIOLENCE', 'ADW', 'CHILD ABUSE', 
        'MURDER', 'ELDER', 'CHLD ABUSE', 'KIDNAPPING', 'FALSE IMPRISONMENT', 'PRVNT WIT', 
        'INFLICT CORPORAL INJURY ON SPOUSE/COHABITANT', 'THREAT', 'THREATEN', 'HARASSMENT']):
        return 'Assault/Violence'

    # Weapons
    elif any(word in charge_desc for word in ['WEAPON', 'FIREARM', 'GUN', 'AMMUNITION', 
        'CONCEALED DIRK', 'METAL KNUCKLES', 'LEADED CANE', 'SWITCHBLADE', 'LRG CAP MAG']):
        return 'Weapons'
        
    # Identity theft / fraud
    elif any(word in charge_desc for word in ['PERSONAL ID INFO', "OTHER'S ID", 'PERSONAL IDENTIFYING INFO', 
        'INSURANCE ENTITLEMENT', 'PERSONATE', 'FORGERY', 'MONEY LAUNDERING', 'ACCESS CARD', 'DFRD', 'FICTITIOUS CHECK',
        'OBTAIN CREDIT', 'USE OTHERS ID', 'PERJURY', 'FORGED INSTRUMENT', 'FALSE STATEMENT', 'FORGE', 'FALSE ENTRIES',
        'DEFRAUD', 'PASS COMPLETED CHECK', 'BAD CHECK', 'BLANK CHECK', 'POSSESS/ETC BAD', 'POSS DL/ID', 'DL/ID', 'ACES CARD']):
        return 'Fraud/Identity Theft'

    # Disorderly Conduct/Public Order
    elif any(word in charge_desc for word in ['DISORDERLY CONDUCT:ALCOHOL', 'TRESPASS', 'POSTED PROPERTY', 
        'LODGE WITHOUT', 'FIGHT/CHALLENGE', 'LOITER', 'TRESP', 'MINOR POSSESS ALCOHOL', 'LIQUOR TO MINOR', 
        'LOUD/UNREASONABLE NOISE', 'LEWD ACT', 'INDECENT EXPOSURE', 'TOUCH PERSON INTIMATELY',
        'PIMPING', 'PANDERING', 'PROTECTIVE ORDER', 'VIOL CRT ORD DOM VIOLENCE', 'CONTEMPT OF COURT']):
        return 'Disorderly Conduct/Public Order'

    # Sex Offense/Registration
    elif any(word in charge_desc for word in ['SEX OFFENDER', '290', 'LEWD', 'LASCIVIOUS', 'CONTACT MINOR WITH INTENT SEX',
        'ANNOY/MOLEST', 'OBSCENE MATTER OF MINOR', 'RAPE', 'ARRANGE/GO TO MEETING', 'L&L', 'SEXUAL GRATIFICATION']):
        return 'Sex Offense/Registration'

    else:
        return 'Other'


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

CATEGORY_SEVERITY_ORDER = [
    "Assault/Violence",
    "Weapons",
    "Sex Offense/Registration",
    "Theft/Burglary/Robbery",
    "Fraud/Identity Theft",
    "Drug Possession/Paraphernalia",
    "DUI",
    "Vandalism",
    "Vehicle-Related (non-DUI)",
    "Obstruct/Resist Officer",
    "Disorderly Conduct/Public Order",
    "Other",
]

STATUTE_SEVERITY_ORDER = ["Felony", "Misdemeanor"]


def assign_primary_charge_by_severity(df):
    """
    Takes in charge-level rows and outputs case-level primary labels
    """

    df = df.copy()

    # Keep only felony/misdemeanor rows
    df = df[df["statute_level"].isin(STATUTE_SEVERITY_ORDER)].copy()

    # Creating dictionaries for severity ranking
    statute_rank_map = {s: i for i, s in enumerate(STATUTE_SEVERITY_ORDER)}
    cat_rank_map = {c: i for i, c in enumerate(CATEGORY_SEVERITY_ORDER)}

    # Assign severity ranks to each charge
    df["_statute_rank"] = df["statute_level"].map(statute_rank_map).fillna(999).astype(int)
    df["_cat_rank"] = df["charge_category"].map(cat_rank_map).fillna(999).astype(int)

    # Count category frequency within case × statute × category
    # Used as a tie breaker -- if two categories have the same statute level, we assign the primary category as the one with more charges in the case
    counts = (
        df.groupby(["source_case_id", "statute_level", "charge_category"], as_index=False)
          .size()
          .rename(columns={"size": "_cat_count"})
    )

    # Add ranks to the aggregated counts table
    counts["_statute_rank"] = counts["statute_level"].map(statute_rank_map).fillna(999).astype(int)
    counts["_cat_rank"] = counts["charge_category"].map(cat_rank_map).fillna(999).astype(int)

    # Determine primary statute level (Felony if any felony exists)
    primary_statute = (
        counts.groupby("source_case_id", as_index=False)["_statute_rank"].min()
              .rename(columns={"_statute_rank": "_primary_statute_rank"})
    )

    inv_statute_rank = {v: k for k, v in statute_rank_map.items()}

    # Results in one row per case, with case id, primary statute rank, and primary statute level
    primary_statute["primary_statute_level"] = primary_statute["_primary_statute_rank"].map(inv_statute_rank)

    # Restrict to rows within primary statute level
    counts = counts.merge(primary_statute[["source_case_id", "_primary_statute_rank"]],
                          on="source_case_id", how="left")

    # Adds each case's primary statute rank to the counts table
    counts = counts[counts["_statute_rank"] == counts["_primary_statute_rank"]].copy()

    # Choose primary charge category within that primary statute level
    counts = counts.sort_values(
        by=["source_case_id", "_cat_rank", "_cat_count", "charge_category"],
        ascending=[True, True, False, True]
    )

    # After sorting, picks the top row per case for the primary charge category
    primary_cat = (
        counts.drop_duplicates("source_case_id", keep="first")
              [["source_case_id", "charge_category"]]
              .rename(columns={"charge_category": "primary_charge_category"})
    )

    # Combine primary statute and primary category back to case level table
    result = primary_statute[["source_case_id", "primary_statute_level"]] \
        .merge(primary_cat, on="source_case_id", how="left")

    # Returns one row per case, with case id, primary statute level (Felony/Misdemeanor), and primary charge category (Assault/Violence, DUI, etc.)
    return result



def enhancement_rates_by_primary_severity(df):
    """
    Returns enhancement rates grouped by:
        primary_charge_category × race_std × Year × primary_statute_level
    """

    df = df.copy()

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