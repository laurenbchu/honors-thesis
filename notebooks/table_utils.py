# ------------------------------------------------------------------
# Table formatting and Latex export utilities
# ------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import re

# --------------------------------------------
# Helper functions
# --------------------------------------------

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


# --------------------------------------------
# Table functions
# --------------------------------------------

def create_reason_for_contact_table(policing_by_reason):
    """
    Create a comprehensive table with stop reason, race, counts, and rates with 95% CIs.
    Each reason for contact is shown only once with race categories as rows beneath it.
    
    Parameters
    ----------
    policing_by_reason : dict
        Dictionary mapping reason_for_contact -> rates DataFrame
        (output from rates_utils.policing_rates_by_reason_for_contact)
    
    Returns
    -------
    DataFrame
        Table with columns: Reason for Contact, Perceived Race, Stops, Searches, 
        Search Rate (%), Hit Rate (%)
    """
    import numpy as np
    
    rows = []
    
    # Define reason order
    reason_order = [
        "Moving violation",
        "Equipment violation",
        "Non-moving violation",
        "Suspect criminal activity"
    ]
    
    for reason in reason_order:
        if reason not in policing_by_reason:
            continue
            
        rates_df = policing_by_reason[reason]
        
        # Pool across all years
        pooled = (
            rates_df.groupby("Perceived Race", as_index=False)
            .agg({
                "Stop Count": "sum",
                "Search Count": "sum",
                "Hit Count": "sum"
            })
        )
        
        # Calculate pooled rates and standard errors
        pooled["Search Rate"] = pooled["Search Count"] / pooled["Stop Count"]
        pooled["Search Rate SE"] = np.sqrt(
            pooled["Search Rate"] * (1 - pooled["Search Rate"]) / pooled["Stop Count"]
        )
        
        pooled["Hit Rate"] = np.where(
            pooled["Search Count"] > 0,
            pooled["Hit Count"] / pooled["Search Count"],
            np.nan
        )
        pooled["Hit Rate SE"] = np.where(
            pooled["Search Count"] > 0,
            np.sqrt(pooled["Hit Rate"] * (1 - pooled["Hit Rate"]) / pooled["Search Count"]),
            np.nan
        )
        
        # Apply race ordering (including "Other")
        race_order_with_other = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
        pooled["Perceived Race"] = pd.Categorical(
            pooled["Perceived Race"], 
            categories=race_order_with_other, 
            ordered=True
        )
        pooled = pooled.sort_values("Perceived Race")
        
        # Convert to percentages and format with CIs
        for idx, row in pooled.iterrows():
            search_rate_pct = row["Search Rate"] * 100
            search_se_pct = row["Search Rate SE"] * 100
            search_ci = 1.96 * search_se_pct
            
            hit_rate_pct = row["Hit Rate"] * 100
            hit_se_pct = row["Hit Rate SE"] * 100
            hit_ci = 1.96 * hit_se_pct
            
            # Only show reason name in the first row for each reason
            reason_display = reason if idx == pooled.index[0] else ""
            
            rows.append({
                "Reason for Contact": reason_display,
                "Perceived Race": row["Perceived Race"],
                "Stops": int(row["Stop Count"]),
                "Searches": int(row["Search Count"]),
                "Search Rate (%)": f"{search_rate_pct:.2f} (±{search_ci:.2f})",
                "Hit Rate (%)": f"{hit_rate_pct:.2f} (±{hit_ci:.2f})" if np.isfinite(hit_rate_pct) else "—"
            })
    
    result = pd.DataFrame(rows)
    
    return result



def agency_black_white_hit_rate_table(df):
    """
    Create a publication-ready table from agency-level Black vs White hit rate comparison.
    """
    
    result = df.copy()
    
    # Select and rename columns
    result = result[[
        "agency_name",
        "White_Search_Count",
        "White_Hit_Count", 
        "White_Hit_Rate",
        "Black_Search_Count",
        "Black_Hit_Count",
        "Black_Hit_Rate"
    ]].copy()
    
    result.columns = [
        "Agency Name",
        "White Search Count",
        "White Hit Count",
        "White Hit Rate",
        "Black Search Count", 
        "Black Hit Count",
        "Black Hit Rate"
    ]
    
    # Format hit rates as percentages with 2 decimal places
    result["White Hit Rate"] = (result["White Hit Rate"] * 100).round(2)
    result["Black Hit Rate"] = (result["Black Hit Rate"] * 100).round(2)
    
    # Sort by agency name
    result = result.sort_values("Agency Name").reset_index(drop=True)
    
    return result



def enhancement_rate_combined_table(enhancement_by_primary):
    """
    Combined enhancement-rate table:
    - Overall enhancement rate by race
    - Enhancement rate by race and statute level
    """

    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Overall", "Misdemeanor", "Felony"]

    # -----------------------------
    # Overall by race
    # -----------------------------
    overall = (
        enhancement_by_primary
        .groupby("race_std", as_index=False)
        .agg({"Enhanced": "sum", "N": "sum"})
    )

    overall["Enhancement Rate"] = overall["Enhanced"] / overall["N"]
    overall["SE"] = np.sqrt(
        overall["Enhancement Rate"] * (1 - overall["Enhancement Rate"]) / overall["N"]
    )

    overall["Enhancement Rate (%)"] = overall.apply(
        lambda row: fmt_est_ci(row["Enhancement Rate"] * 100, row["SE"] * 100, digits=2),
        axis=1
    )

    overall["Statute Level"] = "Overall"

    overall = overall[["race_std", "Statute Level", "N", "Enhancement Rate (%)"]].copy()

    # -----------------------------
    # By statute level
    # -----------------------------
    by_statute = (
        enhancement_by_primary
        .groupby(["race_std", "primary_statute_level"], as_index=False)
        .agg({"Enhanced": "sum", "N": "sum"})
    )

    by_statute["Enhancement Rate"] = by_statute["Enhanced"] / by_statute["N"]
    by_statute["SE"] = np.sqrt(
        by_statute["Enhancement Rate"] * (1 - by_statute["Enhancement Rate"]) / by_statute["N"]
    )

    by_statute["Enhancement Rate (%)"] = by_statute.apply(
        lambda row: fmt_est_ci(row["Enhancement Rate"] * 100, row["SE"] * 100, digits=2),
        axis=1
    )

    by_statute["Statute Level"] = by_statute["primary_statute_level"]

    by_statute = by_statute[["race_std", "Statute Level", "N", "Enhancement Rate (%)"]].copy()

    # -----------------------------
    # Combine
    # -----------------------------
    result = pd.concat([overall, by_statute], ignore_index=True)

    result.columns = ["Canonical Race", "Statute Level", "Number of Cases", "Enhancement Rate (%)"]

    result["race_sort"] = result["Canonical Race"].map({race: i for i, race in enumerate(race_order)})
    result["statute_sort"] = result["Statute Level"].map({stat: i for i, stat in enumerate(statute_order)})

    result = result.sort_values(["race_sort", "statute_sort"]).drop(
        ["race_sort", "statute_sort"], axis=1
    )

    # Blank duplicate race labels within each group
    result["Canonical Race"] = result["Canonical Race"].mask(
        result["Canonical Race"].eq(result["Canonical Race"].shift()), ""
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

    For the charge-category table, categories are excluded if they:
      - have fewer than 500 total wobblers across all races,
      - are the "Other" charge category, or
      - have an overall felony filing rate below 5% when aggregated across all races.
    """

    # Filter to wobbler charges only
    wobblers = df[df["is_wobbler"]].copy()

    # -----------------------------
    # TABLE 1: Overall by race
    # -----------------------------
    wobbler_summary = (
        wobblers
        .groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both columns exist
    for col in ["Felony", "Misdemeanor"]:
        if col not in wobbler_summary.columns:
            wobbler_summary[col] = 0

    wobbler_summary["Total"] = wobbler_summary["Felony"] + wobbler_summary["Misdemeanor"]
    wobbler_summary["Felony Rate"] = wobbler_summary["Felony"] / wobbler_summary["Total"]

    wobbler_summary["Felony Rate SE"] = np.sqrt(
        wobbler_summary["Felony Rate"] *
        (1 - wobbler_summary["Felony Rate"]) /
        wobbler_summary["Total"]
    )

    wobbler_summary.index.name = "Canonical Race"

    wobbler_table = wobbler_summary[
        ["Felony", "Misdemeanor", "Total", "Felony Rate", "Felony Rate SE"]
    ].copy()

    wobbler_table["Felony Rate (%)"] = wobbler_table.apply(
        lambda row: f"{row['Felony Rate']*100:.1f} (±{row['Felony Rate SE']*100*1.96:.1f})",
        axis=1
    )

    wobbler_table = wobbler_table[["Total", "Felony", "Misdemeanor", "Felony Rate (%)"]]
    wobbler_table.columns = [
        "Total Wobblers",
        "Filed as Felony",
        "Filed as Misdemeanor",
        "Felony Rate (%) [95% CI]"
    ]

    # Enforce race order
    wobbler_table = wobbler_table.reindex(TABLE_RACE_ORDER)

    # -----------------------------
    # TABLE 2: By charge category
    # -----------------------------
    wobbler_by_category = (
        wobblers
        .groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["Felony", "Misdemeanor"]:
        if col not in wobbler_by_category.columns:
            wobbler_by_category[col] = 0

    wobbler_by_category["Total"] = (
        wobbler_by_category["Felony"] +
        wobbler_by_category["Misdemeanor"]
    )

    wobbler_by_category["Felony Rate"] = (
        wobbler_by_category["Felony"] /
        wobbler_by_category["Total"]
    )

    wobbler_by_category["Felony Rate SE"] = np.sqrt(
        wobbler_by_category["Felony Rate"] *
        (1 - wobbler_by_category["Felony Rate"]) /
        wobbler_by_category["Total"]
    )

    wobbler_category_table = wobbler_by_category.reset_index()

    # Filter charge categories:
    # - exclude "Other"
    # - exclude categories with <500 total wobblers across all races
    # - exclude categories with overall felony rate <5% across all races
    category_stats = (
        wobbler_category_table
        .groupby("charge_category", as_index=False)
        .agg({
            "Total": "sum",
            "Felony": "sum"
        })
    )

    category_stats["Overall Felony Rate"] = (
        category_stats["Felony"] / category_stats["Total"]
    )

    keep_categories = category_stats.loc[
        (category_stats["charge_category"] != "Other") &
        (category_stats["Total"] >= 500) &
        (category_stats["Overall Felony Rate"] >= 0.05),
        "charge_category"
    ]

    wobbler_category_table = wobbler_category_table[
        wobbler_category_table["charge_category"].isin(keep_categories)
    ].copy()

    wobbler_category_table["Felony Rate (%)"] = wobbler_category_table.apply(
        lambda row: f"{row['Felony Rate']*100:.1f} (±{row['Felony Rate SE']*100*1.96:.1f})",
        axis=1
    )

    wobbler_category_table = wobbler_category_table[[
        "charge_category", "race_std", "Total", "Felony", "Misdemeanor", "Felony Rate (%)"
    ]]

    wobbler_category_table.columns = [
        "Charge Category",
        "Canonical Race",
        "Total Wobblers",
        "Filed as Felony",
        "Filed as Misdemeanor",
        "Felony Rate (%) [95% CI]"
    ]

    # Enforce race order within each category
    wobbler_category_table["Canonical Race"] = pd.Categorical(
        wobbler_category_table["Canonical Race"],
        categories=TABLE_RACE_ORDER,
        ordered=True
    )

    # Sort categories by total wobblers descending across all races
    category_order = (
        wobbler_category_table
        .groupby("Charge Category")["Total Wobblers"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    wobbler_category_table["Charge Category"] = pd.Categorical(
        wobbler_category_table["Charge Category"],
        categories=category_order,
        ordered=True
    )

    wobbler_category_table = wobbler_category_table.sort_values(
        ["Charge Category", "Canonical Race"]
    )

    # Set hierarchical index so category appears once per group
    wobbler_category_table = wobbler_category_table.set_index(
        ["Charge Category", "Canonical Race"]
    )

    return wobbler_table, wobbler_category_table


# --------------------------------------------
# Export latex table functions
# --------------------------------------------

def export_stops_searches_to_latex(summary_df):
    """
    Export Table 1: Policing: Stops and Searches (per 1,000 residents)
    """

    # Format function for estimates with CI
    def fmt_est_ci(est, se, digits=2):
        """Format like: 189.86 (±11.38) with ±1.96 SE"""
        ci = 1.96 * se
        return f"{est:.{digits}f} ({{\\scriptsize $\\pm${ci:.{digits}f}}})"
    
    # Race order for the table
    race_order = [
        "Black/African American",
        "Hispanic/Latino", 
        "White",
        "Asian",
        "Other"
    ]
    
    # Get population (same across years)
    pop_map = summary_df.groupby("Perceived Race")["Population"].first().to_dict()
    
    # Create formatted strings for each metric by race and year
    stops_data = {}
    searches_data = {}
    
    for race in race_order:
        race_df = summary_df[summary_df["Perceived Race"] == race].sort_values("Year")
        
        stops_data[race] = {}
        searches_data[race] = {}
        
        for _, row in race_df.iterrows():
            year = int(row["Year"])
            stops_data[race][year] = fmt_est_ci(
                row["Stops per 1,000"], 
                row["Stops per 1,000 SE"]
            )
            searches_data[race][year] = fmt_est_ci(
                row["Searches per 1,000"],
                row["Searches per 1,000 SE"]
            )
    
    # Build LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Policing: Stops and Searches (per 1,000 residents)}")
    latex.append(r"\label{tab:stops_searches}")
    latex.append(r"\begin{tabular}{l r ccc ccc}")
    latex.append(r"\toprule")
    latex.append(r" & & \multicolumn{3}{c}{Stops per 1,000} & \multicolumn{3}{c}{Searches per 1,000} \\")
    latex.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    latex.append(r"Perceived Race & Population & 2022 & 2023 & 2024 & 2022 & 2023 & 2024 \\")
    latex.append(r"\midrule")
    
    # Add data rows
    for race in race_order:
        pop = int(pop_map[race])
        stops_2022 = stops_data[race].get(2022, "---")
        stops_2023 = stops_data[race].get(2023, "---")
        stops_2024 = stops_data[race].get(2024, "---")
        searches_2022 = searches_data[race].get(2022, "---")
        searches_2023 = searches_data[race].get(2023, "---")
        searches_2024 = searches_data[race].get(2024, "---")
        
        row = f"{race} & {pop:,} & {stops_2022} & {stops_2023} & {stops_2024} & {searches_2022} & {searches_2023} & {searches_2024} \\\\"
        latex.append(row)
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(r"\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses (±1.96 SE).}")
    latex.append(r"\end{table}")
    
    # Write to file
    latex_str = "\n".join(latex)
    with open("../output/tables/stops_searches_per_capita.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print("Table exported: stops_searches_per_capita.tex")
    return latex_str



def export_searches_hits_to_latex(summary_df):
    """
    Export Table 2: Search Rates and Hit Rates by Race and Year
    """
    
    # Format function for percentages with CI
    def fmt_pct_ci(rate, se, digits=2):
        """Format rate as percentage like: 9.93 (±0.58) with ±1.96 SE"""
        pct = rate * 100
        ci = 1.96 * se * 100
        return f"{pct:.{digits}f} ({{\\scriptsize $\\pm${ci:.{digits}f}}})"
    
    # Race order for the table
    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White", 
        "Asian",
        "Other"
    ]
    
    # Create formatted strings for each metric by race and year
    search_rate_data = {}
    hit_rate_data = {}
    
    for race in race_order:
        race_df = summary_df[summary_df["Perceived Race"] == race].sort_values("Year")
        
        search_rate_data[race] = {}
        hit_rate_data[race] = {}
        
        for _, row in race_df.iterrows():
            year = int(row["Year"])
            search_rate_data[race][year] = fmt_pct_ci(
                row["Search Rate"],
                row["Search Rate SE"]
            )
            # Handle NaN hit rates
            if pd.notna(row["Hit Rate"]) and pd.notna(row["Hit Rate SE"]):
                hit_rate_data[race][year] = fmt_pct_ci(
                    row["Hit Rate"],
                    row["Hit Rate SE"]
                )
            else:
                hit_rate_data[race][year] = "---"
    
    # Build LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Policing: Search Rates and Hit Rates by Race and Year}")
    latex.append(r"\label{tab:search_hit_rates}")
    latex.append(r"\begin{tabular}{l ccc ccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{3}{c}{Search Rate (\%)} & \multicolumn{3}{c}{Hit Rate (\%)} \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    latex.append(r"Perceived Race & 2022 & 2023 & 2024 & 2022 & 2023 & 2024 \\")
    latex.append(r"\midrule")
    
    # Add data rows
    for race in race_order:
        search_2022 = search_rate_data[race].get(2022, "---")
        search_2023 = search_rate_data[race].get(2023, "---")
        search_2024 = search_rate_data[race].get(2024, "---")
        hit_2022 = hit_rate_data[race].get(2022, "---")
        hit_2023 = hit_rate_data[race].get(2023, "---")
        hit_2024 = hit_rate_data[race].get(2024, "---")
        
        row = f"{race} & {search_2022} & {search_2023} & {search_2024} & {hit_2022} & {hit_2023} & {hit_2024} \\\\"
        latex.append(row)
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(r"\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses (±1.96 SE).}")
    latex.append(r"\end{table}")
    
    # Write to file
    latex_str = "\n".join(latex)
    with open("../output/tables/searches_hits.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print("Table exported: searches_hits.tex")
    return latex_str



def export_reason_for_contact_table_to_latex(table_df):
    """
    Export reason-for-contact table to LaTeX format.
    
    Parameters
    ----------
    table_df : DataFrame
        Output from create_reason_for_contact_table()
    
    Returns
    -------
    str
        LaTeX table code
    """
    import os
    
    os.makedirs("../output/tables", exist_ok=True)
    
    df = table_df.copy()
    
    # Build LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Policing Outcomes by Reason for Contact and Race (2022-2024 Pooled)}")
    latex.append(r"\label{tab:reason_for_contact}")
    latex.append(r"\begin{tabular}{ll rr ll}")
    latex.append(r"\toprule")
    latex.append(r"Reason for Contact & Perceived Race & Stops & Searches & Search Rate (\%) & Hit Rate (\%) \\")
    latex.append(r"\midrule")
    
    # Add data rows with separators between reason groups
    prev_reason = None
    for idx, row in df.iterrows():
        reason = row["Reason for Contact"]
        race = row["Perceived Race"]
        stops = f"{row['Stops']:,}"
        searches = f"{row['Searches']:,}"
        search_rate = row["Search Rate (%)"]
        hit_rate = row["Hit Rate (%)"]
        
        # Add thin line separator before new reason group (but not before first group)
        if reason and prev_reason is not None:
            latex.append(r"\cmidrule(lr){1-6}")
        
        # Format the line
        if reason:  # First row of a reason group
            latex.append(
                f"{reason} & {race} & {stops} & {searches} & {search_rate} & {hit_rate} \\\\"
            )
            prev_reason = reason
        else:  # Subsequent rows in the same reason group
            latex.append(
                f" & {race} & {stops} & {searches} & {search_rate} & {hit_rate} \\\\"
            )
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} Data pooled across 2022--2024. "
        r"Search rates represent the proportion of stops that resulted in a discretionary search. "
        r"Hit rates represent the proportion of discretionary searches that resulted in contraband "
        r"being found. 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE).}"
    )
    latex.append(r"\end{table}")
    
    # Write to file
    latex_str = "\n".join(latex)
    with open("../output/tables/reason_for_contact.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print("Table exported: reason_for_contact.tex")
    return latex_str
    


def export_agency_hit_rates_to_latex(agency_hit_df):
    """
    Export agency-level Black vs White hit rate comparison table to LaTeX.
    """
    
    os.makedirs("../output/tables", exist_ok=True)
    
    # Work with the original data to calculate standard errors
    df = agency_hit_df.copy()
    
    # Calculate standard errors for hit rates using binomial proportion formula: SE = sqrt(p(1-p)/n)
    df["White_Hit_Rate_SE"] = np.sqrt(
        df["White_Hit_Rate"] * (1 - df["White_Hit_Rate"]) / df["White_Search_Count"]
    )
    
    df["Black_Hit_Rate_SE"] = np.sqrt(
        df["Black_Hit_Rate"] * (1 - df["Black_Hit_Rate"]) / df["Black_Search_Count"]
    )
    
    # Sort by agency name
    df = df.sort_values("agency_name").reset_index(drop=True)
    
    # Build LaTeX table manually for better control
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Agency-Level Comparison: White vs. Black/African American Hit Rates}")
    latex.append(r"\label{tab:agency_hit_rates}")
    latex.append(r"\begin{tabular}{l rr rr}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{2}{c}{White} & \multicolumn{2}{c}{Black/African American} \\")
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    latex.append(r"Agency & Searches & Hit Rate (\%) & Searches & Hit Rate (\%) \\")
    latex.append(r"\midrule")
    
    # Add data rows with hit rates formatted with 95% CI
    for _, row in df.iterrows():
        agency = row["agency_name"]
        
        w_searches = int(row["White_Search_Count"])
        w_rate = row["White_Hit_Rate"] * 100
        w_se = row["White_Hit_Rate_SE"] * 100
        w_ci = 1.96 * w_se
        
        b_searches = int(row["Black_Search_Count"])
        b_rate = row["Black_Hit_Rate"] * 100
        b_se = row["Black_Hit_Rate_SE"] * 100
        b_ci = 1.96 * b_se
        
        latex.append(
            f"{agency} & {w_searches:,} & "
            f"{w_rate:.1f} ({{\\scriptsize $\\pm${w_ci:.1f}}}) & "
            f"{b_searches:,} & "
            f"{b_rate:.1f} ({{\\scriptsize $\\pm${b_ci:.1f}}}) \\\\"
        )
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} Hit rates represent the proportion "
        r"of searches that resulted in contraband being found, with 95\% confidence intervals "
        r"shown in parentheses ($\pm 1.96$ SE). Only agencies with at least 5 searches for both "
        r"White and Black/African American individuals are included.}"
    )
    latex.append(r"\end{table}")
    
    # Write to file
    latex_str = "\n".join(latex)
    with open("../output/tables/agency_hits.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print("Table exported: agency_hits.tex")
    return latex_str



def export_combined_sensitivity_to_latex(
    policing_analysis,
    policing_analysis_mixed,
    policing_analysis_multiperson
):
    """
    Export a combined sensitivity table comparing:
    - Baseline
    - Mixed classification
    - Multiperson stops
    """

    os.makedirs("../output/tables", exist_ok=True)

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian",
        "Other"
    ]

    def fmt_pct_ci(rate, se, digits=2):
        if pd.isna(rate) or pd.isna(se):
            return "---"
        pct = rate * 100
        ci = 1.96 * se * 100
        return f"{pct:.{digits}f} ({{\\scriptsize $\\pm${ci:.{digits}f}}})"

    def build_map(df):
        d = df[df["Year"] == 2024].copy()
        d = d.set_index("Perceived Race")

        out = {}
        for race in race_order:
            if race in d.index:
                row = d.loc[race]
                out[race] = {
                    "Search Rate": fmt_pct_ci(row["Search Rate"], row["Search Rate SE"]),
                    "Hit Rate": fmt_pct_ci(row["Hit Rate"], row["Hit Rate SE"]),
                }
            else:
                out[race] = {"Search Rate": "---", "Hit Rate": "---"}
        return out

    baseline_map = build_map(policing_analysis)
    mixed_map = build_map(policing_analysis_mixed)
    multiperson_map = build_map(policing_analysis_multiperson)

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Sensitivity Analysis: Search and Hit Rates Under Alternative Definitions (2024)}")
    latex.append(r"\label{tab:sensitivity_combined}")
    latex.append(r"\begin{tabular}{l ccc ccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{3}{c}{Search Rate (\%)} & \multicolumn{3}{c}{Hit Rate (\%)} \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    latex.append(r"Perceived Race & Baseline & Mixed & Multiperson & Baseline & Mixed & Multiperson \\")
    latex.append(r"\midrule")

    for race in race_order:
        latex.append(
            f"{race} "
            f"& {baseline_map[race]['Search Rate']} "
            f"& {mixed_map[race]['Search Rate']} "
            f"& {multiperson_map[race]['Search Rate']} "
            f"& {baseline_map[race]['Hit Rate']} "
            f"& {mixed_map[race]['Hit Rate']} "
            f"& {multiperson_map[race]['Hit Rate']} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). Baseline uses the primary specification. Mixed treats searches with mixed or ambiguous legal bases as discretionary. Multiperson includes all people involved in stops rather than only the first listed person.}"
    )
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    with open("../output/tables/sensitivity.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print("Table exported: sensitivity.tex")
    return latex_str



def export_enhancement_tables_to_latex(enhancement_combined, enhancement_category):
    """
    Export the two enhancement tables to LaTeX.

    Outputs:
      - ../output/tables/enhancement_overall.tex
      - ../output/tables/enhancement_by_category.tex
    """

    import os
    import re
    import pandas as pd

    os.makedirs("../output/tables", exist_ok=True)

    def latexify_ci_string(x):
        if pd.isna(x):
            return "---"
        x = str(x)
        return re.sub(r"\(±([^)]+)\)", r"({\\scriptsize $\\pm$\1})", x)

    def fmt_int_with_commas(x):
        if pd.isna(x):
            return "---"
        return f"{int(x):,}"

    # -----------------------------
    # Table 1: Overall + statute level by race
    # -----------------------------
    overall = enhancement_combined.copy()

    overall["Number of Cases"] = overall["Number of Cases"].apply(fmt_int_with_commas)
    overall["Enhancement Rate (%)"] = overall["Enhancement Rate (%)"].apply(latexify_ci_string)

    overall = overall.rename(columns={
        "Enhancement Rate (%)": r"Enhancement Rate (\%)"
    })

    overall_latex = overall.to_latex(
        index=False,
        escape=False,
        column_format="llrr",
        caption="Enhancement Rates by Race and Statute Level",
        label="tab:enhancement_overall",
        na_rep="---"
    )

    # Shade Overall rows
    lines = overall_latex.splitlines()
    shaded_lines = []
    data_started = False

    for line in lines:
        if r"\midrule" in line:
            data_started = True
            shaded_lines.append(line)
            continue

        if data_started and "& Overall &" in line:
            line = r"\rowcolor{gray!15} " + line

        shaded_lines.append(line)

    # Add separators between race groups, but not after the header
    final_lines = []
    for i, line in enumerate(shaded_lines):
        final_lines.append(line)

        if i + 1 < len(shaded_lines):
            next_line = shaded_lines[i + 1]

            current_has_row = "&" in line and r"\bottomrule" not in line and r"\midrule" not in line
            next_is_new_race = "& Overall &" in next_line

            if current_has_row and next_is_new_race:
                final_lines.append(r"\midrule")

    overall_latex = "\n".join(final_lines)

    overall_latex = overall_latex.replace(
        r"\end{tabular}",
        r"""\end{tabular}
\smallskip
\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). The "Overall" row aggregates across statute levels within race.}"""
    )

    with open("../output/tables/enhancement_overall.tex", "w", encoding="utf-8") as f:
        f.write(overall_latex)

    # -----------------------------
    # Table 2: By charge category, race, and statute level
    # -----------------------------
    by_cat = enhancement_category.copy()

    by_cat["Number of Cases"] = by_cat["Number of Cases"].apply(fmt_int_with_commas)
    by_cat["Enhancement Rate (%)"] = by_cat["Enhancement Rate (%)"].apply(latexify_ci_string)

    by_cat = by_cat.rename(columns={
        "Charge Category": "Category",
        "Canonical Race": "Race",
        "Statute Level": "Statute Level",
        "Number of Cases": "Cases",
        "Enhancement Rate (%)": r"Enhancement Rate (\%)"
    })

    # Bold category labels for readability
    by_cat["Category"] = by_cat["Category"].apply(
        lambda x: rf"\textbf{{{x}}}" if x != "" else x
    )

    by_cat_latex = by_cat.to_latex(
        index=False,
        escape=False,
        column_format="lllrr",
        caption="Enhancement Rates by Charge Category, Race, and Statute Level",
        label="tab:enhancement_by_category",
        na_rep="---"
    )

    # Insert separators:
    # - full-width rule between categories
    # - partial rule between races within a category (excluding Category column)
    lines = by_cat_latex.splitlines()
    new_lines = []
    data_started = False

    for i, line in enumerate(lines):
        new_lines.append(line)

        if r"\midrule" in line:
            data_started = True
            continue

        if data_started and i + 1 < len(lines):
            next_line = lines[i + 1]

            current_has_row = "&" in line and r"\bottomrule" not in line
            next_has_row = "&" in next_line and r"\bottomrule" not in next_line

            if current_has_row and next_has_row:
                next_parts = [p.strip() for p in next_line.split("&")]

                next_category = next_parts[0] if len(next_parts) > 0 else ""
                next_race = next_parts[1] if len(next_parts) > 1 else ""

                # New category block
                if next_category != "":
                    new_lines.append(r"\addlinespace")
                    new_lines.append(r"\midrule")

                # New race block within same category:
                # draw a line from Race through Enhancement Rate only
                elif next_race != "":
                    new_lines.append(r"\noalign{\vskip 2pt}")
                    new_lines.append(r"\cline{2-5}")
                    new_lines.append(r"\noalign{\vskip 2pt}")

    by_cat_latex = "\n".join(new_lines)

    by_cat_latex = by_cat_latex.replace(
        r"\end{tabular}",
        r"""\end{tabular}
\smallskip
\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). The "Other" charge category is omitted, and categories shown have an overall enhancement rate of at least 5\%. Categories are ordered by total case volume in descending order.}"""
    )

    with open("../output/tables/enhancement_by_category.tex", "w", encoding="utf-8") as f:
        f.write(by_cat_latex)

    print("Tables exported:")
    print(" - enhancement_overall.tex")
    print(" - enhancement_by_category.tex")

    return overall_latex, by_cat_latex



def export_wobbler_tables_to_latex(wobbler_table, wobbler_category_table):
    """
    Export the two wobbler tables to LaTeX.

    Outputs:
      - ../output/tables/wobbler_overall.tex
      - ../output/tables/wobbler_by_category.tex
    """

    os.makedirs("../output/tables", exist_ok=True)

    def latexify_ci_string(x):
        if pd.isna(x):
            return "---"
        x = str(x)
        return re.sub(r"\(±([^)]+)\)", r"({\\scriptsize $\\pm$\1})", x)

    def fmt_int_with_commas(x):
        if pd.isna(x):
            return "---"
        return f"{int(x):,}"

    # -----------------------------
    # Table 1: Overall by race
    # -----------------------------
    overall = wobbler_table.copy()

    # Add comma formatting to count columns
    for col in ["Total Wobblers", "Filed as Felony", "Filed as Misdemeanor"]:
        overall[col] = overall[col].apply(fmt_int_with_commas)

    overall["Felony Rate (%) [95% CI]"] = overall["Felony Rate (%) [95% CI]"].apply(latexify_ci_string)

    overall = overall.rename(columns={
        "Felony Rate (%) [95% CI]": r"Felony Rate (\%)"
    })

    overall = overall.reset_index()

    overall_latex = overall.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
        caption="Overall Felony Filing Rates for Wobbler Charges by Race",
        label="tab:wobbler_overall",
        na_rep="---"
    )

    # Add note to match your other tables
    overall_latex = overall_latex.replace(
        r"\end{tabular}",
        r"""\end{tabular}
\smallskip
\parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE).}"""
    )

    with open("../output/tables/wobbler_overall.tex", "w", encoding="utf-8") as f:
        f.write(overall_latex)

    # -----------------------------
    # Table 2: By charge category (compact + group lines)
    # -----------------------------
    by_cat = wobbler_category_table.copy().reset_index()

    # Drop "Other" charge category since it is not analyzed elsewhere
    by_cat = by_cat[by_cat["Charge Category"] != "Other"].copy()

    # Compute totals and felony rates by category
    category_stats = (
        by_cat.groupby("Charge Category")
        .agg({
            "Total Wobblers": "sum",
            "Filed as Felony": "sum"
        })
    )

    category_stats["Felony Rate"] = (
        category_stats["Filed as Felony"] /
        category_stats["Total Wobblers"]
    )

    # Keep categories with >500 wobblers AND >=10% felony rate
    category_stats = category_stats[
        (category_stats["Total Wobblers"] > 500) &
        (category_stats["Felony Rate"] >= 0.10)
    ]

    # Sort by total wobblers descending
    category_stats = category_stats.sort_values(
        "Total Wobblers",
        ascending=False
    )

    keep_categories = category_stats.index.tolist()

    by_cat = by_cat[by_cat["Charge Category"].isin(keep_categories)].copy()

    # Sort categories by total wobblers descending, then race order within category
    by_cat["Charge Category"] = pd.Categorical(
        by_cat["Charge Category"],
        categories=keep_categories,
        ordered=True
    )

    by_cat["Canonical Race"] = pd.Categorical(
        by_cat["Canonical Race"],
        categories=TABLE_RACE_ORDER,
        ordered=True
    )

    by_cat = by_cat.sort_values(["Charge Category", "Canonical Race"]).copy()

    # Add comma formatting to count columns
    for col in ["Total Wobblers", "Filed as Felony", "Filed as Misdemeanor"]:
        by_cat[col] = by_cat[col].apply(fmt_int_with_commas)

    by_cat["Felony Rate (%) [95% CI]"] = by_cat["Felony Rate (%) [95% CI]"].apply(latexify_ci_string)

    # Blank repeated category labels so each category prints once
    by_cat["Charge Category"] = by_cat["Charge Category"].astype(object)
    by_cat["Charge Category"] = by_cat["Charge Category"].mask(
        by_cat["Charge Category"].duplicated(), ""
    )

    # Rename columns
    by_cat = by_cat.rename(columns={
        "Charge Category": "Category",
        "Canonical Race": "Race",
        "Total Wobblers": "Total",
        "Filed as Felony": "Felony",
        "Filed as Misdemeanor": "Misdemeanor",
        "Felony Rate (%) [95% CI]": r"Felony Rate (\%)"
    })

    by_cat_latex = by_cat.to_latex(
        index=False,
        escape=False,
        column_format="llrrrr",
        caption="Felony Filing Rates for Wobbler Charges by Charge Category and Race",
        label="tab:wobbler_by_category",
        na_rep="---"
    )

    # Insert horizontal lines between charge-category groups
    lines = by_cat_latex.splitlines()
    new_lines = []
    data_started = False

    for i, line in enumerate(lines):
        new_lines.append(line)

        if r"\midrule" in line:
            data_started = True
            continue

        if data_started and i + 1 < len(lines):
            next_line = lines[i + 1]
            current_has_row = "&" in line and r"\bottomrule" not in line
            next_has_row = "&" in next_line and r"\bottomrule" not in next_line

            if current_has_row and next_has_row:
                first_cell_next = next_line.split("&")[0].strip()
                if first_cell_next != "" and not next_line.startswith(r"\bottomrule"):
                    new_lines.append(r"\addlinespace")
                    new_lines.append(r"\midrule")

    by_cat_latex = "\n".join(new_lines)

    # Add note
    by_cat_latex = by_cat_latex.replace(
        r"\end{tabular}",
        r"""\end{tabular}
    \smallskip
    \parbox{\textwidth}{\footnotesize \textit{Note:} 95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). Categories shown have more than 500 total wobbler charges and an overall felony filing rate of at least 10\%. Categories are ordered by total wobbler volume in descending order.}"""
    )

    with open("../output/tables/wobbler_by_category.tex", "w", encoding="utf-8") as f:
        f.write(by_cat_latex)

    print("Tables exported:")
    print(" - wobbler_overall.tex")
    print(" - wobbler_by_category.tex")

    return overall_latex, by_cat_latex