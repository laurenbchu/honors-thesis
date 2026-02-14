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
    summary["Hits per 1,000"] = (summary["Hit Count"] / summary["Population"]) * per

    return summary


def compute_prosecution_rates(prosecution, convicted_col, enhancement_col):
    """
    Build a race × year prosecution summary table that includes:
    - Total Prosecution Cases
    - Prosecution Rate
    - Enhancement Rate
    """

    # Rename standardized race column
    prosecution = prosecution.rename(columns={"race_std": "Canonical Race"})

    # Conviction / prosecution rate
    conviction_rates = (
        prosecution
        .groupby(["Canonical Race", "Year"])[convicted_col]
        .agg(
            Total_Prosecution_Cases="size",
            Prosecution_Rate="mean"
        )
        .reset_index()
    )

    # Enhancement rate
    enhancement_rates = (
        prosecution
        .groupby(["Canonical Race", "Year"])[enhancement_col]
        .agg(
            Enhancement_Rate="mean"
        )
        .reset_index()
    )

    # Merge both
    prosecution_rates = conviction_rates.merge(
        enhancement_rates,
        on=["Canonical Race", "Year"],
        how="left",
    )

    # Remove underscores for cleaner presentation
    prosecution_rates = prosecution_rates.rename(columns={
        "Total_Prosecution_Cases": "Total Prosecution Cases",
        "Prosecution_Rate": "Prosecution Rate",
        "Enhancement_Rate": "Enhancement Rate",
    })

    return prosecution_rates


def merge_policing_with_prosecution(policing, prosecution):
    """
    Merge policing and prosecution summary tables by race and year.
    """

    analysis = policing.merge(
        prosecution,
        left_on=["Perceived Race", "Year"],
        right_on=["Canonical Race", "Year"],
        how="left",
    )

    # Rename policing race column to "Race"
    analysis = analysis.rename(columns={"Perceived Race": "Race"})

    # Drop duplicate race column from prosecution after merge
    analysis = analysis.drop(columns=["Canonical Race"], errors="ignore")

    return analysis



def add_disparity_metrics(analysis):
    """
    Add White-normalized disparity ratios to the analysis table.

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
        "Prosecution Rate",
        "Enhancement Rate",
    ]

    # 1 is equal to White people
    # Greater than 1 is higher rate than White people
    # Less than 1 is lower rate than White people
    for m in metrics:
        white_ref = (
            out[out["Race"] == "White"]
            .set_index("Year")[m]
        )
        out[f"{m} (White=1)"] = out[m] / out["Year"].map(white_ref)

    return out


def visualization_setup(analysis):
    """
    Setup for keeping race order consistent and saving plots when needed.

    - Keep plots consistent across runs by enforcing a fixed race order
    - Sort by Year then Perceived Race
    """
    out = analysis.copy()

    # Keep plots consistent across runs
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]

    out["Race"] = pd.Categorical(
        out["Race"],
        categories=race_order,
        ordered=True,
    )

    out = out.sort_values(["Year", "Race"])
    return out



def visualize_all_disparity_figures(analysis):
    
    """
    Display the full set of policing + prosecution disparity figures.

    Handles:
      - Counties with 2021–2022 only (e.g., San Mateo)
      - Counties with 2021–2023 (e.g., Orange)
    """

    df = analysis.copy()

    # Determine race order (prefer categorical categories if available)
    if pd.api.types.is_categorical_dtype(df["Race"]):
        races = list(df["Race"].cat.categories)
    else:
        # fallback: deterministic but may not match your preferred ordering
        races = sorted(df["Race"].dropna().unique().tolist())

    years = sorted(df["Year"].dropna().unique().tolist())

    def _set_year_ticks():
        plt.xticks(years)

    def _line_chart(y_col: str, title: str, ylabel: str, caption: str) -> None:
        if y_col not in df.columns:
            return

        plt.figure(figsize=(8, 5))
        for race in races:
            d = df[df["Race"] == race].sort_values("Year")
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


    # --- 3) Prosecution outcomes over time ---
    _line_chart(
        y_col="Conviction Rate",
        title=f"Conviction Rate by Canonical Race ({years[0]}–{years[-1]})",
        ylabel="Conviction Rate",
        caption=(
            f"This figure shows conviction rates for prosecuted cases by racial group from {years[0]} to {years[-1]}. "
            "Rates reflect the proportion of cases resulting in a conviction once charges are filed."
        ),
    )

    _line_chart(
        y_col="Enhancement Rate",
        title=f"Enhancement Rate by Canonical Race ({years[0]}–{years[-1]})",
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
        plot_df = latest_df.set_index("Race")[ratio_cols].reindex(races)

        plot_df.plot(kind="bar", figsize=(10, 4))
        plt.axhline(1.0)
        plt.title(f"Disparity Ratios in Policing and Prosecution ({latest_year}; White = 1)")
        plt.xlabel("Race")
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
        d_earliest = df[df["Year"] == earliest_year].set_index("Race")["Stops per 1,000 (White=1)"]
        d_latest = df[df["Year"] == latest_year].set_index("Race")["Stops per 1,000 (White=1)"]

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


def visualize(analysis):
    """
    Visualize policing + prosecution disparities using the recommended hierarchy:

      A) Exposure (context)
         - Stops per 1,000
         - Searches per 1,000

      B) Policing decision + outcome tests (core)
         - Search Rate (conditional on stop)
         - Hit Rate (OUTCOME TEST; conditional on search)

      C) Prosecution outcome tests (core)
         - Conviction Rate (OUTCOME TEST; conditional on case filed)
         - Enhancement Rate (OUTCOME TEST; conditional on case filed)

      D) Summary comparisons (supporting)
         - White-normalized ratios (latest year), if available
         - Earliest vs latest stop disparity ratio, if available

    """
    import matplotlib.pyplot as plt  # import locally to avoid 'plt' being shadowed elsewhere

    df = analysis.copy()

    # Race order (prefer categorical categories if available)
    if pd.api.types.is_categorical_dtype(df["Race"]):
        races = list(df["Race"].cat.categories)
    else:
        races = sorted(df["Race"].dropna().unique().tolist())

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in analysis['Year'].")

    year_span = f"{years[0]}–{years[-1]}"

    def _set_year_ticks():
        plt.xticks(years)

    def _line_chart(
        y_col: str,
        *,
        title: str,
        subtitle: str | None,
        ylabel: str,
        caption: str,
    ) -> None:
        if y_col not in df.columns:
            return

        plt.figure(figsize=(8, 5))

        for race in races:
            d = df[df["Race"] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                plt.plot(d["Year"], d[y_col], marker="o", label=race)

        # Title + short subtitle line (when applicable)
        if subtitle:
            plt.suptitle(title, y=1.02)
            plt.title(subtitle, fontsize=10)
        else:
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

    def _bar_chart_latest_year(
        cols: list[str],
        *,
        title: str,
        subtitle: str | None,
        ylabel: str,
        caption: str,
    ) -> None:
        latest_year = years[-1]
        latest_df = df[df["Year"] == latest_year].copy()
        if latest_df.empty:
            return

        cols = [c for c in cols if c in latest_df.columns]
        if not cols:
            return

        plot_df = latest_df.set_index("Race")[cols].reindex(races)

        ax = plot_df.plot(kind="bar", figsize=(10, 4))
        ax.axhline(1.0)

        if subtitle:
            plt.suptitle(title, y=1.02)
            plt.title(subtitle, fontsize=10)
        else:
            plt.title(title)

        plt.xlabel("Perceived Race")
        plt.ylabel(ylabel)

        plt.figtext(
            0.5, -0.08,
            caption,
            ha="center",
            fontsize=10,
            wrap=True
        )

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def _bar_chart_earliest_vs_latest(
        col: str,
        *,
        title: str,
        subtitle: str | None,
        ylabel: str,
        caption: str,
    ) -> None:
        if col not in df.columns:
            return
        if len(years) < 2:
            return

        earliest_year = years[0]
        latest_year = years[-1]
        if earliest_year == latest_year:
            return

        d_earliest = df[df["Year"] == earliest_year].set_index("Perceived Race")[col]
        d_latest = df[df["Year"] == latest_year].set_index("Perceived Race")[col]

        comp = pd.DataFrame({str(earliest_year): d_earliest, str(latest_year): d_latest}).reindex(races)

        ax = comp.plot(kind="bar", figsize=(8, 4))
        ax.axhline(1.0)

        if subtitle:
            plt.suptitle(title, y=1.02)
            plt.title(subtitle, fontsize=10)
        else:
            plt.title(title)

        plt.xlabel("Perceived Race")
        plt.ylabel(ylabel)

        plt.figtext(
            0.5, -0.08,
            caption,
            ha="center",
            fontsize=10,
            wrap=True
        )

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # A) Exposure (context)
    # =========================================================
    _line_chart(
        "Stops per 1,000",
        title=f"Stops per 1,000 Residents by Perceived Race ({year_span})",
        subtitle="Exposure (context): population-normalized stop frequency",
        ylabel="Stops per 1,000 (2020 Census baseline)",
        caption=(
            f"This figure shows how often people in each racial group are stopped overall from {years[0]} to {years[-1]}. "
            "Rates are expressed as stops per 1,000 residents, which adjusts for differences in population sizes."
        ),
    )

    _line_chart(
        "Searches per 1,000",
        title=f"Searches per 1,000 Residents by Perceived Race ({year_span})",
        subtitle="Exposure (context): population-normalized search frequency",
        ylabel="Searches per 1,000 (2020 Census baseline)",
        caption=(
            f"This figure shows how often people in each racial group are searched overall from {years[0]} to {years[-1]}. "
            "Searches are reported per 1,000 residents, reflecting population-level exposure to police searches."
        ),
    )

    # =========================================================
    # B) Policing decision + outcome tests (core)
    # =========================================================
    _line_chart(
        "Search Rate",
        title=f"Search Rate by Perceived Race ({year_span})",
        subtitle="Conditional on stop: searches ÷ stops",
        ylabel="Search Rate",
        caption=(
            f"This figure shows the likelihood that a police stop results in a search for each racial group from {years[0]} to {years[-1]}. "
            "The search rate reflects police decision-making conditional on a stop, rather than overall exposure."
        ),
    )

    _line_chart(
        "Hit Rate",
        title=f"Hit Rate by Perceived Race ({year_span})",
        subtitle="Outcome test (conditional on search): hits ÷ searches",
        ylabel="Hit Rate",
        caption=(
            f"This figure shows how often police searches yield contraband for each racial group from {years[0]} to {years[-1]}. "
            "Because it is conditional on searches, the hit rate functions as an outcome test of whether searches are equally productive across groups."
        ),
    )

    # =========================================================
    # C) Prosecution outcome tests (core)
    # =========================================================
    _line_chart(
        "Conviction Rate",
        title=f"Conviction Rate by Perceived Race ({year_span})",
        subtitle="Outcome test (conditional on case filed): convictions ÷ cases",
        ylabel="Conviction Rate",
        caption=(
            f"This figure shows conviction rates for prosecuted cases by racial group from {years[0]} to {years[-1]}. "
            "Rates reflect the proportion of cases resulting in a conviction once charges are filed."
        ),
    )

    _line_chart(
        "Enhancement Rate",
        title=f"Enhancement Rate by Perceived Race ({year_span})",
        subtitle="Outcome test (conditional on case filed): enhancements ÷ cases",
        ylabel="Enhancement Rate",
        caption=(
            f"This figure shows how often cases include sentencing enhancements by racial group from {years[0]} to {years[-1]}. "
            "Enhancement rates reflect prosecutorial decisions following case filing."
        ),
    )

    # =========================================================
    # D) Summary comparisons (supporting)
    # =========================================================
    _bar_chart_latest_year(
        cols=[
            "Stops per 1,000 (White=1)",
            "Searches per 1,000 (White=1)",
            "Conviction Rate (White=1)",
            "Enhancement Rate (White=1)",
        ],
        title=f"Disparity Ratios in Policing and Prosecution ({years[-1]}; White = 1)",
        subtitle="Summary comparison: each metric normalized to White residents (White = 1)",
        ylabel="Ratio relative to White",
        caption=(
            f"This figure compares policing and prosecution outcomes across racial groups in {years[-1]}, normalized to White residents "
            "(White = 1). Values greater than 1 indicate higher rates relative to White residents, while values below 1 indicate lower rates."
        ),
    )

    _bar_chart_earliest_vs_latest(
        col="Stops per 1,000 (White=1)",
        title=f"Stop Rate Disparity Ratio Change ({years[0]} vs {years[-1]}; White = 1)",
        subtitle="Summary comparison over time: earliest vs latest year available",
        ylabel="Ratio relative to White",
        caption=(
            f"This figure compares stop rate disparities in {years[0]} and {years[-1]} using White residents as the reference group (White = 1). "
            "Changes over time indicate whether racial disparities in police stops widened or narrowed."
        ),
    )