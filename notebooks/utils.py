import pandas as pd


def load_ripa_policing_data(county, years=(2021, 2022, 2023)):
    """
    Load cleaned RIPA policing data for a given county and multiple years,
    add a YEAR column, concatenate, and map race codes to labels.

    Parameters
    ----------
    county : str
        County name used in filenames (e.g. "orange", "sanmateo")
    years : iterable of int, default (2021, 2022, 2023)
        Years of data to load

    Returns
    -------
    policing : pandas.DataFrame
        Concatenated RIPA policing dataframe with YEAR column
        and mapped RAE_FULL race labels
    """

    base_url = (
        "https://raw.githubusercontent.com/"
        "laurenbchu/honors-thesis/main/data/cleaned/"
    )

    dfs = []

    for year in years:
        url = f"{base_url}cleaned_ripa_{county}_{year}.csv"
        df = pd.read_csv(url)
        df["YEAR"] = year
        dfs.append(df)

    policing = pd.concat(dfs, axis=0, ignore_index=True)

    # Map race codes to race labels
    races = {
        1: "Asian",
        2: "Black/African American",
        3: "Hispanic/Latino",
        4: "Middle Eastern/South Asian",
        5: "Native American",
        6: "Pacific Islander",
        7: "White",
        8: "Multiracial",
    }

    policing["RAE_FULL"] = policing["RAE_FULL"].map(races)

    return policing


def load_census(county_fips):
    """
    Load and relabel Decennial Census PL Table P2 data for a given county.

    Parameters
    ----------
    county_fips : str
        5-digit county FIPS code (e.g. "06059" for Orange County,
        "06081" for San Mateo County)

    Returns
    -------
    census_relabeled : pandas.DataFrame
        Census P2 table with readable column labels and NA columns removed
    """

    # ---- Fetch P2 data ----
    data_url = (
        f"https://api.census.gov/data/2020/dec/pl"
        f"?get=group(P2)&ucgid=0500000US{county_fips}"
    )
    census = pd.read_json(data_url)
    census.columns = census.iloc[0]
    census = census.iloc[1:]

    # ---- Fetch metadata for variable labels ----
    metadata_url = "https://api.census.gov/data/2020/dec/pl/variables.json"
    metadata = pd.read_json(metadata_url, typ="series")

    variables = metadata["variables"]
    meta = pd.DataFrame.from_dict(variables, orient="index")
    code_to_label = meta["label"].to_dict()

    # ---- Clean + relabel ----
    census = census.loc[:, ~census.columns.str.endswith("NA")]
    census_relabeled = census.rename(columns=code_to_label)

    return census_relabeled


def compute_prosecution_rates(
    prosecution,
    race_col="canonical_race",
    year_col="year",
    convicted_col="was_convicted",
    enhancement_col="is_enhancement_charge",
):
    conviction_rates = (
        prosecution
        .groupby([race_col, year_col])[convicted_col]
        .agg(total_cases="count", conviction_rate="mean")
        .reset_index()
    )

    enhancement_rates = (
        prosecution
        .groupby([race_col, year_col])[enhancement_col]
        .agg(total_cases="count", enhancement_rate="mean")
        .reset_index()
    )

    prosecution_rates = conviction_rates.merge(
        enhancement_rates[[race_col, year_col, "enhancement_rate"]],
        on=[race_col, year_col],
        how="left",
    )

    return prosecution_rates
