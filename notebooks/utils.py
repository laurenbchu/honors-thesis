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


def policing_summary_table(df, census):
    """
    Build a race × year policing summary table that includes:
    - Stop Count
    - Search Count
    - Hit Count
    - Search Rate
    - Hit Rate
    - Stops per 1,000
    - Searches per 1,000
    - Hits per 1,000
    """

    # Group by year and race
    g = df.groupby(["year", "race_std"])

    summary = g.agg(
        Stop_Count=("action_any_search", "size"),
        Search_Count=("action_any_search", "sum"),
        Hit_Count=("contraband_any", "sum"),
    ).reset_index()

    # Rename columns
    summary = summary.rename(columns={
        "year": "Year",
        "race_std": "Perceived Race",
        "Stop_Count": "Stop Count",
        "Search_Count": "Search Count",
        "Hit_Count": "Hit Count",
    })

    # Conditional rates
    summary["Search Rate"] = summary["Search Count"] / summary["Stop Count"]
    summary["Hit Rate"] = summary["Hit Count"] / summary["Search Count"]

    # Add population
    summary["Population"] = summary["Perceived Race"].map(census)

    # Per-capita rates (per 1,000 residents)
    per = 1000
    summary["Stops per 1,000"] = (summary["Stop Count"] / summary["Population"]) * per
    summary["Searches per 1,000"] = (summary["Search Count"] / summary["Population"]) * per

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

    summary = (
        df.groupby(["Canonical Race", "Year", "statute_level"])
          .agg(
              **{
                  "Total Charges": ("was_filed_by_da", "size"),
                  "Charge Rate": ("was_filed_by_da", "mean"),
                  "Enhancement Rate": ("is_enhancement_charge", "mean"),
              }
          )
          .reset_index()
    )

    return summary


def add_white_normalized_ratios(df, race_col="Perceived Race", year_col="Year", metrics=None, strata_cols=None):
    """
    Add White-normalized disparity ratios to any table.

    If you have multiple rows per year (e.g., statute_level), pass strata_cols so
    White is matched within the same Year + strata group.

    Example:
        add_white_normalized_ratios(prosecution_analysis,
                                    race_col="Canonical Race",
                                    strata_cols=["statute_level"],
                                    metrics=["Charge Rate","Enhancement Rate"])
    """

    out = df.copy()

    if strata_cols is None:
        strata_cols = []

    key_cols = [year_col] + list(strata_cols)

    # If metrics not provided, pick all numeric columns except keys + race
    if metrics is None:
        metrics = [
            c for c in out.columns
            if c not in ([race_col] + key_cols)
            and pd.api.types.is_numeric_dtype(out[c])
        ]

    # White reference table: one row per (Year + strata) with White's metric value
    white_ref = out[out[race_col] == "White"][key_cols + metrics].copy()

    # Rename metric columns so we can merge them back cleanly
    rename_map = {m: f"{m} (White)" for m in metrics}
    white_ref = white_ref.rename(columns=rename_map)

    # Merge White reference values back onto everyone by Year (+ strata)
    out = out.merge(white_ref, on=key_cols, how="left")

    # Compute ratios
    for m in metrics:
        out[f"{m} (White=1)"] = out[m] / out[f"{m} (White)"]

    # Optional: drop the intermediate "(White)" columns
    out = out.drop(columns=[f"{m} (White)" for m in metrics], errors="ignore")

    return out

import pandas as pd
import matplotlib.pyplot as plt

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
    """
    Policing visuals (most relevant):
      1) Stops per 1,000 by Perceived Race (exposure)
      2) Searches per 1,000 by Perceived Race (exposure)
      3) Search Rate by Perceived Race (decision)
      4) Hit Rate by Perceived Race (outcome test)
      5) Latest-year disparity ratios (White=1) for Stops/Searches/Search Rate/Hit Rate (summary)
    """
    df = policing_analysis.copy()
    race_col = "Perceived Race"

    df = visualization_setup(df, race_col=race_col)

    if pd.api.types.is_categorical_dtype(df[race_col]):
        races = list(df[race_col].cat.categories)
    else:
        races = sorted(df[race_col].dropna().unique().tolist())

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in policing_analysis['Year'].")

    def _set_year_ticks():
        plt.xticks(years)

    def _line_chart(y_col, title, ylabel, caption):
        if y_col not in df.columns:
            return

        plt.figure(figsize=(8, 5))
        for race in races:
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                plt.plot(d["Year"], d[y_col], marker="o", label=race)

        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.legend(title="Perceived Race")
        _set_year_ticks()

        plt.figtext(0.5, -0.05, caption, ha="center", fontsize=10, wrap=True)
        plt.tight_layout()
        plt.show()

    # --- Core policing figures ---
    _line_chart(
        "Stops per 1,000",
        title=f"Stops per 1,000 Residents by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Stops per 1,000 (2020 Census baseline)",
        caption="Population-normalized exposure to police stops over time."
    )

    _line_chart(
        "Searches per 1,000",
        title=f"Searches per 1,000 Residents by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Searches per 1,000 (2020 Census baseline)",
        caption="Population-normalized exposure to police searches over time."
    )

    _line_chart(
        "Search Rate",
        title=f"Search Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Search Rate",
        caption="Conditional on being stopped: how often stops turn into searches."
    )

    _line_chart(
        "Hit Rate",
        title=f"Hit Rate by Perceived Race ({years[0]}–{years[-1]})",
        ylabel="Hit Rate",
        caption="Outcome test (conditional on search): how often searches yield contraband."
    )

    # --- Summary: disparity ratios (latest year) ---
    latest_year = years[-1]
    latest_df = df[df["Year"] == latest_year].copy()

    ratio_cols = [
        "Stops per 1,000 (White=1)",
        "Searches per 1,000 (White=1)",
        "Search Rate (White=1)",
        "Hit Rate (White=1)",
    ]
    ratio_cols = [c for c in ratio_cols if c in latest_df.columns]

    if not latest_df.empty and ratio_cols:
        plot_df = latest_df.set_index(race_col)[ratio_cols].reindex(races)

        plot_df.plot(kind="bar", figsize=(10, 4))
        plt.axhline(1.0)
        plt.title(f"Policing Disparity Ratios ({latest_year}; White = 1)")
        plt.xlabel("Race")
        plt.ylabel("Ratio relative to White")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()


def visualize_prosecution(prosecution_analysis):
    """
    Prosecution visuals (most relevant):
      For EACH statute_level (Felony/Misdemeanor/Infraction):
        1) Charge Rate by Canonical Race over time
        2) Enhancement Rate by Canonical Race over time
        3) Latest-year disparity ratios (White=1) for Charge/Enhancement (summary)
    """
    df = prosecution_analysis.copy()
    race_col = "Canonical Race"

    df = visualization_setup(df, race_col=race_col)

    if pd.api.types.is_categorical_dtype(df[race_col]):
        races = list(df[race_col].cat.categories)
    else:
        races = sorted(df[race_col].dropna().unique().tolist())

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in prosecution_analysis['Year'].")

    statute_levels = sorted(df["statute_level"].dropna().unique().tolist())
    if not statute_levels:
        raise ValueError("No statute_level values found in prosecution_analysis['statute_level'].")

    def _set_year_ticks():
        plt.xticks(years)

    def _line_chart(dsub, y_col, title, ylabel, caption):
        if y_col not in dsub.columns:
            return

        plt.figure(figsize=(8, 5))
        for race in races:
            d = dsub[dsub[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                plt.plot(d["Year"], d[y_col], marker="o", label=race)

        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.legend(title="Race")  # IMPORTANT: not "Perceived Race"
        _set_year_ticks()

        plt.figtext(0.5, -0.05, caption, ha="center", fontsize=10, wrap=True)
        plt.tight_layout()
        plt.show()

    for lvl in statute_levels:
        dsub = df[df["statute_level"] == lvl].copy()
        if dsub.empty:
            continue

        _line_chart(
            dsub,
            "Charge Rate",
            title=f"Charge Rate by Race ({lvl}; {years[0]}–{years[-1]})",
            ylabel="Charge Rate",
            caption=f"Charge outcomes restricted to {lvl} charges."
        )

        _line_chart(
            dsub,
            "Enhancement Rate",
            title=f"Enhancement Rate by Race ({lvl}; {years[0]}–{years[-1]})",
            ylabel="Enhancement Rate",
            caption=f"Enhancement outcomes restricted to {lvl} charges."
        )

        # Latest-year ratios for this statute level
        latest_year = years[-1]
        latest_df = dsub[dsub["Year"] == latest_year].copy()

        ratio_cols = [
            "Charge Rate (White=1)",
            "Enhancement Rate (White=1)",
        ]
        ratio_cols = [c for c in ratio_cols if c in latest_df.columns]

        if not latest_df.empty and ratio_cols:
            plot_df = latest_df.set_index(race_col)[ratio_cols].reindex(races)

            plot_df.plot(kind="bar", figsize=(8, 4))
            plt.axhline(1.0)
            plt.title(f"Prosecution Disparity Ratios ({lvl}, {latest_year}; White = 1)")
            plt.xlabel("Race")  # IMPORTANT: x-axis is Race
            plt.ylabel("Ratio relative to White")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.show()