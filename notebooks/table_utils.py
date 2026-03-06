# ------------------------------------------------------------------
# Table formatting and Latex export utilities
# ------------------------------------------------------------------

import numpy as np
import pandas as pd

TABLE_RACE_ORDER = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]


def fmt_est_ci(est, se, digits=2):
    """Format like: 189.86 (±11.38)"""
    return f"{est:.{digits}f} (±{(1.96*se):.{digits}f})"



def make_pivot_with_ci(df, value_col, se_col):
    """
    Returns a wide table with one column per year containing formatted estimate (±CI).
    """
    tmp = df[["Perceived Race", "Year", value_col, se_col]].copy()
    tmp["cell"] = [fmt_est_ci(e, s, digits=2) for e, s in zip(tmp[value_col], tmp[se_col])]
    wide = tmp.pivot(index="Perceived Race", columns="Year", values="cell")
    # order columns if desired
    wide = wide.reindex(columns=[y for y in (2022, 2023, 2024) if y in wide.columns])
    return wide



def visualization_setup(df, race_col):
    """
    Enforce a consistent race order and sort for plotting.
    race_col should be either:
      - "Perceived Race" (policing)
      - "Canonical Race" (prosecution)
    """
    out = df.copy()

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian"]

    if race_col in out.columns:
        out[race_col] = pd.Categorical(out[race_col], categories=race_order, ordered=True)

    if "Year" in out.columns and race_col in out.columns:
        out = out.sort_values(["Year", race_col])

    return out



def enhancement_rate_race_table(enhancement_by_primary):
    """
    Overall enhancement rates by race (aggregating across all statute levels and charge categories).
    """
    summary = (
        enhancement_by_primary
        .groupby('race_std', as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[['race_std', 'N', 'Enhancement Rate (%)']].copy()
    result.columns = ['Canonical Race', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by race order
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result = result.sort_values('race_sort').drop('race_sort', axis=1)
    
    return result.reset_index(drop=True)



def enhancement_rate_race_statute_table(enhancement_by_primary):
    """
    Enhancement rates by race, stratified by statute level.
    """
    summary = (
        enhancement_by_primary
        .groupby(['race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI using the standard format
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[['race_std', 'primary_statute_level', 'N', 'Enhancement Rate (%)']].copy()
    result.columns = ['Canonical Race', 'Statute Level', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by race order and statute level
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Misdemeanor", "Felony"]
    
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result['statute_sort'] = result['Statute Level'].map({stat: i for i, stat in enumerate(statute_order)})
    result = result.sort_values(['race_sort', 'statute_sort']).drop(['race_sort', 'statute_sort'], axis=1)
    
    # Blank out duplicate race names (show race only once per group)
    result['Canonical Race'] = result['Canonical Race'].mask(
        result['Canonical Race'].eq(result['Canonical Race'].shift()), ''
    )
    
    return result.reset_index(drop=True)



def enhancement_rate_race_statute_category_table(enhancement_by_primary, top_categories=None):
    """
    Enhancement rates by race, stratified by both statute level AND charge category.
    """
    df = enhancement_by_primary.copy()
    
    # Filter to specific categories if provided
    if top_categories is not None:
        df = df[df['primary_charge_category'].isin(top_categories)]
    
    summary = (
        df.groupby(['primary_charge_category', 'race_std', 'primary_statute_level'], as_index=False)
        .agg({'Enhanced': 'sum', 'N': 'sum'})
    )
    
    # Calculate enhancement rate and SE (as proportions)
    summary['Enhancement Rate'] = summary['Enhanced'] / summary['N']
    summary['SE'] = np.sqrt(
        summary['Enhancement Rate'] * (1 - summary['Enhancement Rate']) / summary['N']
    )
    
    # Convert to percentage and format with CI
    summary['Enhancement Rate (%)'] = summary.apply(
        lambda row: fmt_est_ci(row['Enhancement Rate'] * 100, row['SE'] * 100, digits=2),
        axis=1
    )
    
    # Select and rename columns for display
    result = summary[[
        'primary_charge_category', 'race_std', 'primary_statute_level', 'N', 'Enhancement Rate (%)'
    ]].copy()
    result.columns = ['Charge Category', 'Canonical Race', 'Statute Level', 'Number of Cases', 'Enhancement Rate (%)']
    
    # Sort by category (by total N), then race, then statute level
    category_order = (
        summary.groupby('primary_charge_category')['N']
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Misdemeanor", "Felony"]
    
    result['cat_sort'] = result['Charge Category'].map({cat: i for i, cat in enumerate(category_order)})
    result['race_sort'] = result['Canonical Race'].map({race: i for i, race in enumerate(race_order)})
    result['statute_sort'] = result['Statute Level'].map({stat: i for i, stat in enumerate(statute_order)})
    result = result.sort_values(['cat_sort', 'race_sort', 'statute_sort']).drop(['cat_sort', 'race_sort', 'statute_sort'], axis=1)
    
    # Blank out duplicate charge category and race names for readability
    result['Charge Category'] = result['Charge Category'].mask(
        result['Charge Category'].eq(result['Charge Category'].shift()), ''
    )
    result['Canonical Race'] = result['Canonical Race'].mask(
        result['Canonical Race'].eq(result['Canonical Race'].shift()), ''
    )
    
    return result.reset_index(drop=True)



def wobbler_felony_rate_tables(df):
    """
    Generate two tables for wobbler charges:
      1) Overall felony filing rates for wobblers by race
      2) Felony filing rates for wobblers by charge category and race
    """

    import numpy as np
    import pandas as pd

    # Filter to wobbler charges only
    wobblers = df[df['is_wobbler']].copy()

    # -----------------------------
    # TABLE 1: Overall by race
    # -----------------------------
    wobbler_summary = (
        wobblers
        .groupby(['race_std', 'statute_level'])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both columns exist
    for col in ['Felony', 'Misdemeanor']:
        if col not in wobbler_summary.columns:
            wobbler_summary[col] = 0

    wobbler_summary['Total'] = wobbler_summary['Felony'] + wobbler_summary['Misdemeanor']
    wobbler_summary['Felony Rate'] = wobbler_summary['Felony'] / wobbler_summary['Total']

    wobbler_summary['Felony Rate SE'] = np.sqrt(
        wobbler_summary['Felony Rate'] *
        (1 - wobbler_summary['Felony Rate']) /
        wobbler_summary['Total']
    )

    wobbler_summary.index.name = 'Canonical Race'

    wobbler_table = wobbler_summary[['Felony', 'Misdemeanor', 'Total', 'Felony Rate', 'Felony Rate SE']].copy()

    wobbler_table['Felony Rate (%)'] = wobbler_table.apply(
        lambda row: f"{row['Felony Rate']*100:.1f} (±{row['Felony Rate SE']*100*1.96:.1f})",
        axis=1
    )

    wobbler_table = wobbler_table[['Total', 'Felony', 'Misdemeanor', 'Felony Rate (%)']]
    wobbler_table.columns = [
        'Total Wobblers',
        'Filed as Felony',
        'Filed as Misdemeanor',
        'Felony Rate (%) [95% CI]'
    ]

    # Enforce race order
    wobbler_table = wobbler_table.reindex(TABLE_RACE_ORDER)

    # -----------------------------
    # TABLE 2: By charge category
    # -----------------------------
    wobbler_by_category = (
        wobblers
        .groupby(['charge_category', 'race_std', 'statute_level'])
        .size()
        .unstack(fill_value=0)
    )

    for col in ['Felony', 'Misdemeanor']:
        if col not in wobbler_by_category.columns:
            wobbler_by_category[col] = 0

    wobbler_by_category['Total'] = (
        wobbler_by_category['Felony'] +
        wobbler_by_category['Misdemeanor']
    )

    wobbler_by_category['Felony Rate'] = (
        wobbler_by_category['Felony'] /
        wobbler_by_category['Total']
    )

    wobbler_by_category['Felony Rate SE'] = np.sqrt(
        wobbler_by_category['Felony Rate'] *
        (1 - wobbler_by_category['Felony Rate']) /
        wobbler_by_category['Total']
    )

    wobbler_category_table = wobbler_by_category.reset_index()

    wobbler_category_table['Felony Rate (%)'] = wobbler_category_table.apply(
        lambda row: f"{row['Felony Rate']*100:.1f} (±{row['Felony Rate SE']*100*1.96:.1f})",
        axis=1
    )

    wobbler_category_table = wobbler_category_table[[
        'charge_category', 'race_std', 'Total', 'Felony', 'Misdemeanor', 'Felony Rate (%)'
    ]]

    wobbler_category_table.columns = [
        'Charge Category',
        'Canonical Race',
        'Total Wobblers',
        'Filed as Felony',
        'Filed as Misdemeanor',
        'Felony Rate (%) [95% CI]'
    ]

    # Enforce race order within each category
    wobbler_category_table['Canonical Race'] = pd.Categorical(
        wobbler_category_table['Canonical Race'],
        categories=TABLE_RACE_ORDER,
        ordered=True
    )

    wobbler_category_table = wobbler_category_table.sort_values(
    ['Charge Category', 'Canonical Race']
    )

    # Set hierarchical index so category appears once per group
    wobbler_category_table = wobbler_category_table.set_index(
        ['Charge Category', 'Canonical Race']
    )

    return wobbler_table, wobbler_category_table



def wobbler_felony_rate_by_category(df):
    # Filter to wobbler charges only
    wobblers = df[df["is_wobbler"]].copy()

    # Aggregate by charge category and statute level (across all races)
    category_summary = (
        wobblers
        .groupby(["charge_category", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both statute level columns exist
    for col in ["Felony", "Misdemeanor"]:
        if col not in category_summary.columns:
            category_summary[col] = 0

    # Totals
    category_summary["Total"] = category_summary["Felony"] + category_summary["Misdemeanor"]

    # Remove categories with < 500 wobblers AND remove "Other"
    category_summary = category_summary[
        (category_summary["Total"] >= 500) &
        (category_summary.index != "Other")
    ].copy()

    # Felony rate (MUST exist before SE)
    category_summary["Felony Rate"] = np.where(
        category_summary["Total"] > 0,
        category_summary["Felony"] / category_summary["Total"],
        np.nan
    )

    # Remove categories with felony rate under 50%
    category_summary = category_summary[category_summary["Felony Rate"] >= 0.50]

    # Standard error for 95% CI
    category_summary["Felony Rate SE"] = np.sqrt(
        category_summary["Felony Rate"] *
        (1 - category_summary["Felony Rate"]) /
        category_summary["Total"]
    )

    # Sort by felony rate (descending)
    category_summary = category_summary.sort_values("Felony Rate", ascending=False)

    # Format output
    category_summary["Felony Rate (%)"] = category_summary.apply(
        lambda row: f"{row['Felony Rate']*100:.1f} (±{row['Felony Rate SE']*100*1.96:.1f})",
        axis=1
    )

    result = category_summary[["Total", "Felony", "Misdemeanor", "Felony Rate (%)"]].copy()
    result.columns = [
        "Total Wobblers",
        "Filed as Felony",
        "Filed as Misdemeanor",
        "Felony Rate (%) [95% CI]"
    ]
    result.index.name = "Charge Category"
    return result
