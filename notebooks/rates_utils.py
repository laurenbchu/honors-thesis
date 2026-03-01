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


def enhancement_rates_by_category(df):
    """
    Compute enhancement rates by category x race x year x statute
    Among cases in category X, what fraction had any enhancement alleged?
    """
    # Cases are assigned to their modal (most common) substantive charge category
    case_level = (df
                .groupby(['source_case_id', 'race_std', 'Year'])
                .agg(
                    any_enhancement=('any_enhancement_in_case', 'max'),
                    # define a case "primary" category/statute for stratification
                    charge_category=('charge_category', lambda x: x.value_counts().index[0]),
                    statute_level=('statute_level', lambda x: x.value_counts().index[0])
                )
                .reset_index()
                )

    g = (case_level.groupby(['charge_category', 'race_std', 'Year', 'statute_level'])
        .agg(Enhanced=('any_enhancement','sum'),
            N=('any_enhancement','size'))
        .reset_index())

    g['Enhancement Rate'] = g['Enhanced'] / g['N']

    # SE + 95% CI (Wald). For small N, consider Wilson instead.
    g['SE'] = np.sqrt(g['Enhancement Rate'] * (1 - g['Enhancement Rate']) / g['N'])
    g['CI Lower'] = np.maximum(0, g['Enhancement Rate'] - 1.96 * g['SE'])
    g['CI Upper'] = np.minimum(1, g['Enhancement Rate'] + 1.96 * g['SE'])
    return g