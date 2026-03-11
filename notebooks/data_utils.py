# -------------------------------------------
# Data loading and preprocessing utilities
# -------------------------------------------

import pandas as pd
import numpy as np

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