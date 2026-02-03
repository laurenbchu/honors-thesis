import pandas as pd
import matplotlib.pyplot as plt

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


def condense_ripa_races(
    df: pd.DataFrame,
    race_col: str = "RAE_FULL",
    to_other: list = None,
    other_label: str = "Other",
):
    """
    Collapse selected RIPA race categories into a single 'Other' category.

    Parameters
    ----------
    df : pandas.DataFrame
        RIPA policing dataframe
    race_col : str, default "RAE_FULL"
        Column containing perceived race labels
    to_other : list of str, optional
        Race categories to collapse into 'Other'
        Defaults to standard RIPA categories
    other_label : str, default "Other"
        Label used for collapsed categories

    Returns
    -------
    pandas.DataFrame
        Copy of df with race categories collapsed
    """

    if to_other is None:
        to_other = [
            "Middle Eastern/South Asian",
            "Multiracial",
            "Native American",
            "Pacific Islander",
        ]

    out = df.copy()
    out[race_col] = out[race_col].replace(to_other, other_label)

    return out


def add_search_and_hit_indicators(
    df: pd.DataFrame,
    contraband_cols: list = None,
):
    """
    Add binary indicators for whether a search occurred and whether a hit occurred
    in RIPA policing data.

    Parameters
    ----------
    df : pandas.DataFrame
        RIPA policing dataframe
    contraband_cols : list of str, optional
        Columns indicating types of contraband found

    Returns
    -------
    pandas.DataFrame
        Copy of df with SEARCHED and HIT columns added
    """

    if contraband_cols is None:
        contraband_cols = [
            "CED_FIREARM",
            "CED_AMMUNITION",
            "CED_WEAPON",
            "CED_DRUGS",
            "CED_ALCOHOL",
            "CED_MONEY",
            "CED_DRUG_PARAPHERNALIA",
            "CED_STOLEN_PROP",
            "CED_ELECT_DEVICE",
            "CED_OTHER_CONTRABAND",
        ]

    out = df.copy()

    # Any search occurred
    out["SEARCHED"] = (
        (out["ADS_SEARCH_PERSON"] == 1)
        | (out["ADS_SEARCH_PROPERTY"] == 1)
    ).astype(int)

    # Any contraband found
    out["HIT"] = (out[contraband_cols].sum(axis=1) > 0).astype(int)

    return out


def policing_table_for_year(
    policing: pd.DataFrame,
    year: int
):
    """
    Build a race-grouped policing summary table for one year.
    """
    d = policing[policing["YEAR"] == year]
    g = d.groupby("RAE_FULL")

    out = pd.DataFrame({
        "Search Rate": g["SEARCHED"].mean(),
        "Hit Rate": g["HIT"].mean(),
        "Search Count": g["SEARCHED"].sum(),
        "Hit Count": g["HIT"].sum(),
        "Stop Count": g.size(),
    })

    out.index.name = "Perceived Race"
    return out


def policing_tables_by_year(
    policing: pd.DataFrame,
    years: list
):
    """
    Build policing tables for multiple years. Returns {year: table}.
    """
    return {
        yr: policing_table_for_year(
            policing,
            yr
        )
        for yr in years
    }


def census_rollup(
    census_relabeled: pd.DataFrame,
    id_cols: list = None
):
    """
    Convert a relabeled Decennial PL P2 census table to your coarse race population Series.

    This function:
      1) coerces all non-id columns to numeric
      2) selects the row at row_idx (default 0)
      3) returns a coarse population Series with keys:
         ["Hispanic/Latino", "White", "Black/African American", "Asian", "Other"]

    Parameters
    ----------
    census_relabeled : pandas.DataFrame
        Census P2 table with readable labels (after renaming columns)
    id_cols : list of str, optional
        Columns to keep as non-numeric identifiers

    Returns
    -------
    pandas.Series
        Coarse race populations
    """

    if id_cols is None:
        id_cols = ["NAME", "Geography", "Uniform Census Geography Identifier clause"]

    df = census_relabeled.copy()

    # Coerce to numeric
    for c in df.columns:
        if c not in id_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pop = df.iloc[0]

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


def add_per_capita_rates(
    policing_table: pd.DataFrame,
    census_coarse: pd.Series,
    per: float = 1000.0,
    search_count_col: str = "Search Count",
    hit_count_col: str = "Hit Count",
    stop_count_col: str = "Stop Count",
):
    """
    Add per-capita columns (per N people) to a policing table indexed by race.
    """
    out = policing_table.copy()

    # align population to table index (Perceived Race)
    pop = census_coarse.reindex(out.index)

    out[f"Searches per {int(per):,}"] = (out[search_count_col] / pop) * per
    out[f"Hits per {int(per):,}"] = (out[hit_count_col] / pop) * per
    out[f"Stops per {int(per):,}"] = (out[stop_count_col] / pop) * per

    return out


def add_per_capita_rates_by_year(
    tables_by_year: dict[int, pd.DataFrame],
    census_coarse: pd.Series,
    per: float = 1000.0
):
    """
    Apply per-capita columns to each year's policing table.
    """
    return {yr: add_per_capita_rates(t, census_coarse, per=per) for yr, t in tables_by_year.items()}


def concat_policing_tables(
    tables_by_year: dict[int, pd.DataFrame],
    year_col_name: str = "Year"
):
    """
    Convert {year: table} into one long dataframe with a Year column.
    """
    frames = []
    for yr, t in sorted(tables_by_year.items()):
        frames.append(t.reset_index().assign(**{year_col_name: yr}))
    return pd.concat(frames, ignore_index=True)


def merge_policing_with_prosecution(
    policing_all: pd.DataFrame,
    prosecution_rates: pd.DataFrame,
    pros_race_col: str = "canonical_race",
    pros_year_col: str = "year",
    race_mapping: dict = None
):
    """
    Merge policing and prosecution tables using standardized policing columns.

    Assumes policing_all has:
      - 'Perceived Race'
      - 'Year'
    """

    if race_mapping is None:
        race_mapping = {
            "Latinx": "Hispanic/Latino",
            "Black": "Black/African American",
        }

    pros = prosecution_rates.copy()
    pros["police_equivalent_race"] = pros[pros_race_col].replace(race_mapping)

    analysis = policing_all.merge(
        pros,
        left_on=["Perceived Race", "Year"],
        right_on=["police_equivalent_race", pros_year_col],
        how="left",
    ).rename(columns={
        "total_cases": "Total Prosecution Cases",
        pros_race_col: "Canonical Prosecution Race",
        "conviction_rate": "Conviction Rate",
        "enhancement_rate": "Enhancement Rate",
    })

    # Only drop helper columns — never drop policing 'Year'
    drop_cols = ["police_equivalent_race"]
    if pros_year_col != "Year":
        drop_cols.append(pros_year_col)

    analysis = analysis.drop(columns=drop_cols, errors="ignore")

    return analysis


def add_disparity_metrics(analysis: pd.DataFrame):
    """
    Add White-normalized disparity ratios and the Search–Hit Gap to the analysis table.

    This function assumes:
      - policing columns are standardized
      - race column is 'Perceived Race'
      - year column is 'Year'
      - White is the reference group
      - all metric column names are fixed

    Returns
    -------
    pandas.DataFrame
        Copy of analysis with disparity metrics added
    """

    out = analysis.copy()

    # Create White-normalized disparity ratios
    # "How high is this rate for a given group relative to White people?"
    metrics = [
        "Stops per 1,000",
        "Searches per 1,000",
        "Search Rate",
        "Hit Rate",
        "Conviction Rate",
        "Enhancement Rate",
    ]

    # 1 is equal to White people
    # Greater than 1 is higher rate than White people
    # Less than 1 is lower rate than White people
    for m in metrics:
        white_ref = (
            out[out["Perceived Race"] == "White"]
            .set_index("Year")[m]
        )
        out[f"{m} (White=1)"] = out[m] / out["Year"].map(white_ref)

    # "How much searching occurs relative to how often searches pay off?"
    # Larger gap is many searches with fewer hits
    # Compare gaps across races, not absolute values
    out["Search–Hit Gap"] = out["Search Rate"] - out["Hit Rate"]

    return out


def visualization_setup(analysis: pd.DataFrame):
    """
    Setup for keeping race order consistent and saving plots when needed.

    - Keep plots consistent across runs by enforcing a fixed race order
    - Sort by Year then Perceived Race
    """
    out = analysis.copy()

    # Keep plots consistent across runs
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]

    out["Perceived Race"] = pd.Categorical(
        out["Perceived Race"],
        categories=race_order,
        ordered=True,
    )

    out = out.sort_values(["Year", "Perceived Race"])
    return out



def visualize_all_disparity_figures(analysis: pd.DataFrame):
    
    """
    Display the full set of policing + prosecution disparity figures.

    Assumptions:
      - analysis has 'Year' and 'Perceived Race'
      - 'Perceived Race' is a categorical with your desired ordering
        (run visualization_setup(analysis) first)
      - Displays plots only (no saving)

    Handles:
      - Counties with 2021–2022 only (e.g., San Mateo)
      - Counties with 2021–2023 (e.g., Orange)
    """

    df = analysis.copy()

    # Determine race order (prefer categorical categories if available)
    if pd.api.types.is_categorical_dtype(df["Perceived Race"]):
        races = list(df["Perceived Race"].cat.categories)
    else:
        # fallback: deterministic but may not match your preferred ordering
        races = sorted(df["Perceived Race"].dropna().unique().tolist())

    years = sorted(df["Year"].dropna().unique().tolist())

    def _set_year_ticks():
        plt.xticks(years)

    def _line_chart(y_col: str, title: str, ylabel: str, caption: str) -> None:
        if y_col not in df.columns:
            return

        plt.figure(figsize=(8, 5))
        for race in races:
            d = df[df["Perceived Race"] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                plt.plot(d["Year"], d[y_col], marker="o", label=race)

        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.legend()
        _set_year_ticks()

        plt.figtext(
            0.5, -0.05,
            caption,
            ha="center",
            fontsize=10,
            wrap=True
        )

        plt.tight_layout()
        plt.show()

    # --- 1) Exposure over time (per-capita) ---
    _line_chart(
        y_col="Stops per 1,000",
        title=f"Stops per 1,000 Residents by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Stops per 1,000 (2020 Census baseline)",
        caption=(
            f"This figure shows how often people in each racial group are stopped overall from {years[0]} to {years[-1]}. "
            "Rates are expressed as stops per 1,000 residents, which adjusts for differences in population sizes."
        ),
    )

    _line_chart(
        y_col="Searches per 1,000",
        title=f"Searches per 1,000 Residents by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Searches per 1,000 (2020 Census baseline)",
        caption=(
            f"This figure shows how often people in each racial group are searched overall from {years[0]} to {years[-1]}. "
            "Searches are reported per 1,000 residents, reflecting population-level exposure to police searches."
        ),
    )

    # --- 2) Policing decision-making (conditional rates) ---
    _line_chart(
        y_col="Search Rate",
        title=f"Search Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Search Rate",
        caption=(
            f"This figure shows the likelihood that a police stop results in a search for each racial group from {years[0]} to {years[-1]}. "
            "The search rate reflects police decision-making conditional on a stop, rather than overall exposure."
        ),
    )

    _line_chart(
        y_col="Hit Rate",
        title=f"Hit Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Hit Rate",
        caption=(
            f"This figure shows how often police stops result in contraband being found for each racial group from {years[0]} to {years[-1]}. "
            "The hit rate reflects the frequency of successful searches given police contact."
        ),
    )

    _line_chart(
        y_col="Search–Hit Gap",
        title=f"Search–Hit Gap (Search Rate − Hit Rate) by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Search–Hit Gap",
        caption=(
            f"This figure shows the difference between search rates and hit rates for each racial group from {years[0]} to {years[-1]}. "
            "Larger gaps indicate more frequent searching relative to successful contraband recovery."
        ),
    )

    # --- 3) Prosecution outcomes over time ---
    _line_chart(
        y_col="Conviction Rate",
        title=f"Conviction Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Conviction Rate",
        caption=(
            f"This figure shows conviction rates for prosecuted cases by racial group from {years[0]} to {years[-1]}. "
            "Rates reflect the proportion of cases resulting in a conviction once charges are filed."
        ),
    )

    _line_chart(
        y_col="Enhancement Rate",
        title=f"Enhancement Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Enhancement Rate",
        caption=(
            f"This figure shows how often cases include sentencing enhancements by racial group from {years[0]} to {years[-1]}. "
            "Enhancement rates reflect prosecutorial decisions following case filing."
        ),
    )

    # --- 4) Headline disparities: White-normalized ratios (latest year) ---
    latest_year = years[-1]
    latest_df = df[df["Year"] == latest_year].copy()
    ratio_cols = [
        "Stops per 1,000 (White=1)",
        "Searches per 1,000 (White=1)",
        "Conviction Rate (White=1)",
        "Enhancement Rate (White=1)",
    ]
    ratio_cols = [c for c in ratio_cols if c in latest_df.columns]

    if len(latest_df) > 0 and ratio_cols:
        plot_df = latest_df.set_index("Perceived Race")[ratio_cols].reindex(races)

        plot_df.plot(kind="bar", figsize=(10, 4))
        plt.axhline(1.0)
        plt.title(f"Disparity Ratios in Policing and Prosecution ({latest_year}; White = 1)")
        plt.xlabel("Perceived Race")
        plt.ylabel("Ratio relative to White")

        plt.figtext(
            0.5, -0.08,
            (
                f"This figure compares policing and prosecution outcomes across racial groups in {latest_year}, normalized to White residents "
                "(White = 1). Values greater than 1 indicate higher rates relative to White residents, while values below 1 indicate lower rates."
            ),
            ha="center",
            fontsize=10,
            wrap=True
        )

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # --- 5) Earliest vs latest comparison (only if both years exist and the ratio column exists) ---
    earliest_year = years[0]
    if earliest_year != latest_year and "Stops per 1,000 (White=1)" in df.columns:
        d_earliest = df[df["Year"] == earliest_year].set_index("Perceived Race")["Stops per 1,000 (White=1)"]
        d_latest = df[df["Year"] == latest_year].set_index("Perceived Race")["Stops per 1,000 (White=1)"]

        comp = pd.DataFrame({str(earliest_year): d_earliest, str(latest_year): d_latest}).reindex(races)

        comp.plot(kind="bar", figsize=(8, 4))
        plt.axhline(1.0)
        plt.title(f"Stop Rate Disparity Ratio Change ({earliest_year} vs {latest_year}; White = 1)")
        plt.xlabel("Perceived Race")
        plt.ylabel("Ratio relative to White")

        plt.figtext(
            0.5, -0.08,
            (
                f"This figure compares stop rate disparities in {earliest_year} and {latest_year} using White residents as the reference group (White = 1). "
                "Changes over time indicate whether racial disparities in police stops widened or narrowed."
            ),
            ha="center",
            fontsize=10,
            wrap=True
        )

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
