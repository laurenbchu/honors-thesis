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
    """
    
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



def agency_black_white_rates_table(df):
    """
    Create a publication-ready display table from agency-level
    Black vs White search rate and hit rate comparison.
    Rates are formatted as percentages with 95% CIs.
    """

    result = df.copy()

    def fmt_rate_ci(rate, se):
        """Format rate as 'XX.X (±YY.Y)'"""
        if pd.isna(rate) or pd.isna(se):
            return "---"
        return f"{rate*100:.1f} (±{1.96*se*100:.1f})"

    result["White Search Rate"] = result.apply(
        lambda r: fmt_rate_ci(r["White_Search_Rate"], r["White_Search_Rate_SE"]), axis=1
    )
    result["Black Search Rate"] = result.apply(
        lambda r: fmt_rate_ci(r["Black_Search_Rate"], r["Black_Search_Rate_SE"]), axis=1
    )
    result["White Hit Rate"] = result.apply(
        lambda r: fmt_rate_ci(r["White_Hit_Rate"], r["White_Hit_Rate_SE"]), axis=1
    )
    result["Black Hit Rate"] = result.apply(
        lambda r: fmt_rate_ci(r["Black_Hit_Rate"], r["Black_Hit_Rate_SE"]), axis=1
    )

    result = result[[
        "agency_name",
        "White_Stop_Count",  "White_Search_Count", "White Search Rate", "White Hit Rate",
        "Black_Stop_Count",  "Black_Search_Count", "Black Search Rate", "Black Hit Rate",
    ]].copy()

    result.columns = [
        "Agency",
        "White Stops", "White Searches", "White Search Rate (%)", "White Hit Rate (%)",
        "Black Stops", "Black Searches", "Black Search Rate (%)", "Black Hit Rate (%)",
    ]

    result = result.sort_values("Agency").reset_index(drop=True)

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
    
    race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
    statute_order = ["Misdemeanor", "Felony"]
    category_order = ["Assault/Violence", "Weapons", "Obstruct/Resist Officer"]
    
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
    Generate a single combined wobbler table with:
      - An "Overall" group showing felony filing rates by race across all wobblers
      - Category-specific groups showing felony filing rates by race and charge category

    Categories are excluded if they:
      - have fewer than 500 total wobblers across all races,
      - are the "Other" charge category, or
      - have an overall felony filing rate below 10%.
    """

    def compute_rates(grouped_df):
        """Given a df with Felony and Misdemeanor columns, compute rate and SE."""
        grouped_df = grouped_df.copy()
        for col in ["Felony", "Misdemeanor"]:
            if col not in grouped_df.columns:
                grouped_df[col] = 0
        grouped_df["Total"] = grouped_df["Felony"] + grouped_df["Misdemeanor"]
        grouped_df["Felony Rate"] = grouped_df["Felony"] / grouped_df["Total"]
        grouped_df["Felony Rate SE"] = np.sqrt(
            grouped_df["Felony Rate"] *
            (1 - grouped_df["Felony Rate"]) /
            grouped_df["Total"]
        )
        grouped_df["Felony Rate (%) [95% CI]"] = grouped_df.apply(
            lambda row: (
                f"{row['Felony Rate']*100:.1f} "
                f"(\u00b1{row['Felony Rate SE']*100*1.96:.1f})"
            ),
            axis=1,
        )
        return grouped_df

    wobblers = df[df["is_wobbler"]].copy()

    # ------------------------------------------------------------------
    # OVERALL section: one row per race
    # ------------------------------------------------------------------
    overall_raw = (
        wobblers
        .groupby(["race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    overall_raw = compute_rates(overall_raw)
    overall_raw["Charge Category"] = "Overall"
    overall_raw = overall_raw.rename(columns={"race_std": "Canonical Race"})
    overall_raw = overall_raw[[
        "Charge Category", "Canonical Race",
        "Total", "Felony", "Misdemeanor", "Felony Rate (%) [95% CI]"
    ]]

    # ------------------------------------------------------------------
    # BY CATEGORY section
    # ------------------------------------------------------------------
    cat_raw = (
        wobblers
        .groupby(["charge_category", "race_std", "statute_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cat_raw = compute_rates(cat_raw)
    cat_raw = cat_raw.rename(columns={
        "charge_category": "Charge Category",
        "race_std":        "Canonical Race",
    })
    cat_raw = cat_raw[[
        "Charge Category", "Canonical Race",
        "Total", "Felony", "Misdemeanor", "Felony Rate (%) [95% CI]"
    ]]

    # Filter categories
    category_stats = (
        cat_raw.groupby("Charge Category", as_index=False)
        .agg(Total=("Total", "sum"), Felony=("Felony", "sum"))
    )
    category_stats["Overall Rate"] = (
        category_stats["Felony"] / category_stats["Total"]
    )
    keep_categories = category_stats.loc[
        (category_stats["Charge Category"] != "Other") &
        (category_stats["Total"] >= 500) &
        (category_stats["Overall Rate"] >= 0.1),
        "Charge Category",
    ].tolist()

    cat_raw = cat_raw[cat_raw["Charge Category"].isin(keep_categories)].copy()

    # Sort categories by total wobblers descending
    category_order = (
        cat_raw.groupby("Charge Category")["Total"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ------------------------------------------------------------------
    # Combine: Overall first, then categories
    # ------------------------------------------------------------------
    combined = pd.concat([overall_raw, cat_raw], ignore_index=True)

    # Enforce category order with Overall first
    combined["Charge Category"] = pd.Categorical(
        combined["Charge Category"],
        categories=["Overall"] + category_order,
        ordered=True,
    )

    # Enforce race order within each category
    combined["Canonical Race"] = pd.Categorical(
        combined["Canonical Race"],
        categories=TABLE_RACE_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(["Charge Category", "Canonical Race"])
    combined.columns = [
        "Charge Category",
        "Canonical Race",
        "Total Wobblers",
        "Filed as Felony",
        "Filed as Misdemeanor",
        "Felony Rate (%) [95% CI]",
    ]

    # Hierarchical index so category appears once per group
    combined = combined.set_index(["Charge Category", "Canonical Race"])

    return combined

# --------------------------------------------
# Export latex table functions
# --------------------------------------------

def export_policing_pooled_to_latex(policing_analysis):
    """
    Export pooled 2022-2024 policing summary table to LaTeX.

    Columns: Population, Stops per 1,000, Searches per 1,000,
             Search Rate (% with 95% CI), Hit Rate (% with 95% CI).

    Stops and searches per 1,000 are means across years (no CI).
    Search and hit rates are pooled from raw counts (CI shown).

    Outputs:
      - ../output/tables/policing_pooled.tex
    """

    os.makedirs("../output/tables", exist_ok=True)

    def latexify_ci_string(x):
        if pd.isna(x):
            return "---"
        return re.sub(r"\(±([^)]+)\)", r"({\\scriptsize $\\pm$\1})", str(x))

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian",
        "Other",
    ]

    df = policing_analysis.copy()

    # Population (fixed across years)
    pop = (
        df[["Perceived Race", "Population"]]
        .drop_duplicates("Perceived Race")
        .set_index("Perceived Race")
        .reindex(race_order)["Population"]
        .apply(lambda x: f"{int(x):,}")
    )

    # Stops and searches per 1,000: mean across years, no CI
    stops = (
        df.groupby("Perceived Race")["Stops per 1,000"]
        .mean()
        .reindex(race_order)
        .apply(lambda x: f"{x:.2f}")
    )

    searches = (
        df.groupby("Perceived Race")["Searches per 1,000"]
        .mean()
        .reindex(race_order)
        .apply(lambda x: f"{x:.2f}")
    )

    # Search rate: pool raw counts, compute rate + SE
    sc = (
        df.groupby("Perceived Race")[["Search Count", "Stop Count"]]
        .sum()
        .reindex(race_order)
    )
    sc["rate"] = sc["Search Count"] / sc["Stop Count"]
    sc["se"]   = np.sqrt(sc["rate"] * (1 - sc["rate"]) / sc["Stop Count"])
    search_rate = sc.apply(
        lambda row: (
            f"{row['rate']*100:.2f} "
            f"(±{row['se']*100*1.96:.2f})"
        ),
        axis=1,
    ).apply(latexify_ci_string)

    # Hit rate: pool raw counts, compute rate + SE
    hc = (
        df.groupby("Perceived Race")[["Hit Count", "Search Count"]]
        .sum()
        .reindex(race_order)
    )
    hc["rate"] = hc["Hit Count"] / hc["Search Count"]
    hc["se"]   = np.sqrt(hc["rate"] * (1 - hc["rate"]) / hc["Search Count"])
    hit_rate = hc.apply(
        lambda row: (
            f"{row['rate']*100:.2f} "
            f"(±{row['se']*100*1.96:.2f})"
        ),
        axis=1,
    ).apply(latexify_ci_string)

    # ------------------------------------------------------------------
    # Build LaTeX
    # ------------------------------------------------------------------
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{Pooled Policing Outcomes by Perceived Race, "
        r"Orange County 2022--2024}"
    )
    latex.append(r"\label{tab:policing_pooled}")
    latex.append(r"\begin{tabular}{l r cc cc}")
    latex.append(r"\toprule")
    latex.append(
        r" & & \multicolumn{2}{c}{Per capita (per 1,000)} & "
        r"\multicolumn{2}{c}{Conditional rates (\%)} \\"
    )
    latex.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    latex.append(
        r"Perceived Race & Population & "
        r"Stops & Searches & "
        r"Search Rate & Hit Rate \\"
    )
    latex.append(r"\midrule")

    for race in race_order:
        latex.append(
            f"{race} & {pop[race]} & "
            f"{stops[race]} & {searches[race]} & "
            f"{search_rate[race]} & {hit_rate[race]} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} "
        r"Stops and searches per 1,000 residents are averaged across "
        r"2022--2024 using 2020 Census population denominators and are "
        r"reported without uncertainty bands. "
        r"Search and hit rates are computed from pooled counts across all "
        r"three years. "
        r"95\% confidence intervals shown in parentheses ($\pm$1.96 SE).}"
    )
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    with open("../output/tables/policing_pooled.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print("Table exported: policing_pooled.tex")
    return latex_str



def export_policing_by_year(policing_analysis):
    """
    Export merged Table 1+2: Policing summary by race and year.
    Combines stops per 1,000, searches per 1,000, search rate, and hit rate
    into a single landscape table.
    """

    def fmt_est_ci(est, se, digits=2):
        """Format estimate with ±1.96 SE confidence interval."""
        if pd.isna(est) or pd.isna(se):
            return "---"
        ci = 1.96 * se
        return f"{est:.{digits}f} {{\\scriptsize ($\\pm${ci:.{digits}f})}}"

    def fmt_pct_ci(rate, se, digits=2):
        """Format rate as percentage with ±1.96 SE confidence interval."""
        if pd.isna(rate) or pd.isna(se):
            return "---"
        pct = rate * 100
        ci = 1.96 * se * 100
        return f"{pct:.{digits}f} {{\\scriptsize ($\\pm${ci:.{digits}f})}}"

    race_order = [
        "Black/African American",
        "Hispanic/Latino",
        "White",
        "Asian",
        "Other",
    ]

    years = sorted(policing_analysis["Year"].dropna().unique().astype(int).tolist())

    # Build one row per race
    rows = []
    for race in race_order:
        d = policing_analysis[policing_analysis["Perceived Race"] == race].sort_values("Year")
        pop = int(d["Population"].iloc[0]) if not d.empty else 0

        row = {"Perceived Race": race, "Population": f"{pop:,}"}

        for yr in years:
            yr_d = d[d["Year"] == yr]
            if yr_d.empty:
                row[f"stops_{yr}"]  = "---"
                row[f"srch_{yr}"]   = "---"
                row[f"sr_{yr}"]     = "---"
                row[f"hr_{yr}"]     = "---"
            else:
                r = yr_d.iloc[0]
                row[f"stops_{yr}"] = f"{r['Stops per 1,000']:.2f}"
                row[f"srch_{yr}"]  = f"{r['Searches per 1,000']:.2f}"
                row[f"sr_{yr}"] = fmt_pct_ci(
                    r["Search Rate"], r["Search Rate SE"]
                )
                hr_val = r["Hit Rate"]   if "Hit Rate"    in r.index else float("nan")
                hr_se  = r["Hit Rate SE"] if "Hit Rate SE" in r.index else float("nan")
                row[f"hr_{yr}"] = fmt_pct_ci(hr_val, hr_se)

        rows.append(row)

    # ------------------------------------------------------------------
    # Build LaTeX
    # ------------------------------------------------------------------
    n_years = len(years)
    year_str = " & ".join(str(y) for y in years)

    # Column spec: race label | population | 3 cols each for 4 metrics
    col_spec = "l r " + " ".join(["ccc"] * 4)

    latex = []
    latex.append(r"\begin{landscape}")
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{Policing Outcomes by Perceived Race and Year, Orange County 2022--2024}"
    )
    latex.append(r"\label{tab:policing_summary}")
    latex.append(rf"\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Row 1: metric group headers
    latex.append(
        r" & & "
        r"\multicolumn{3}{c}{Stops per 1,000} & "
        r"\multicolumn{3}{c}{Searches per 1,000} & "
        r"\multicolumn{3}{c}{Search Rate (\%)} & "
        r"\multicolumn{3}{c}{Hit Rate (\%)} \\"
    )

    # Cmidrule separators under each metric group
    latex.append(
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} "
        r"\cmidrule(lr){9-11} \cmidrule(lr){12-14}"
    )

    # Row 2: year sub-headers
    latex.append(
        rf"Perceived Race & Population & "
        rf"{year_str} & {year_str} & {year_str} & {year_str} \\"
    )
    latex.append(r"\midrule")

    # Data rows
    for row in rows:
        race = row["Perceived Race"]
        pop  = row["Population"]
        cells = []
        for yr in years:
            cells.append(row[f"stops_{yr}"])
        for yr in years:
            cells.append(row[f"srch_{yr}"])
        for yr in years:
            cells.append(row[f"sr_{yr}"])
        for yr in years:
            cells.append(row[f"hr_{yr}"])
        latex.append(f"{race} & {pop} & " + " & ".join(cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\linewidth}{\footnotesize \textit{Note:} "
        r"Stops and searches per 1,000 residents use 2020 Census population denominators. "
        r"95\% confidence intervals shown in parentheses ($\pm$1.96 SE) for conditional "
        r"rates; per-capita rates are reported without uncertainty bands.}"
    )
    latex.append(r"\end{table}")
    latex.append(r"\end{landscape}")

    latex_str = "\n".join(latex)

    with open("../output/tables/policing_by_year.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print("Table exported: policing_by_year.tex")
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
    


def export_agency_hit_rates_to_latex(publication_table, agency_hit_summary):
    """
    Export agency-level Black vs White search rate and hit rate
    comparison table to LaTeX. Accepts the output of
    agency_black_white_rates_table() and agency_hit_summary for
    computing exclusion flags.

    Agencies excluded from the figure (Black searches < 30 or
    Black hit rate CI > ±20pp) are flagged with a dagger.
    """

    os.makedirs("../output/tables", exist_ok=True)

    df = publication_table.copy()

    # ── Compute exclusion flags from raw summary ──────────────────────
    raw = agency_hit_summary.copy()
    raw["Black_Hit_Rate_SE"] = np.sqrt(
        raw["Black_Hit_Rate"] *
        (1 - raw["Black_Hit_Rate"]) /
        raw["Black_Search_Count"]
    )
    raw["Black_CI_Width"] = raw["Black_Hit_Rate_SE"] * 1.96 * 100
    raw["Excluded"] = (
        (raw["Black_Search_Count"] < 30) |
        (raw["Black_CI_Width"] > 20)
    )

    exclusion_map = raw.set_index("agency_name")["Excluded"].to_dict()
    df["Excluded"] = df["Agency"].map(exclusion_map).fillna(False)

    # ── Sort alphabetically ───────────────────────────────────────────
    df = df.sort_values("Agency").reset_index(drop=True)

    # ── Build LaTeX ───────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{Agency-Level Comparison: White vs.\ "
        r"Black/African American Search and Hit Rates}"
    )
    latex.append(r"\label{tab:agency_hit_rates}")
    latex.append(r"\begin{tabular}{l rrrr rrrr}")
    latex.append(r"\toprule")
    latex.append(
        r" & \multicolumn{4}{c}{White} & "
        r"\multicolumn{4}{c}{Black/African American} \\"
    )
    latex.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    latex.append(
        r"Agency & Stops & Searches & Search Rate (\%) & Hit Rate (\%) & "
        r"Stops & Searches & Search Rate (\%) & Hit Rate (\%) \\"
    )
    latex.append(r"\midrule")

    for _, row in df.iterrows():
        dagger = r"\textsuperscript{\dag}" if row["Excluded"] else ""
        agency = f"{row['Agency']}{dagger}"

        # Parse formatted rate strings like "7.6 (±0.4)"
        def parse_rate(s):
            """Extract point estimate and CI from '7.6 (±0.4)' format."""
            if pd.isna(s) or s == "---":
                return "---"
            return s.replace("(±", r"({\scriptsize $\pm$").replace(")", "})")

        latex.append(
            f"{agency} & "
            f"{int(row['White Stops']):,} & "
            f"{int(row['White Searches']):,} & "
            f"{parse_rate(row['White Search Rate (%)'])} & "
            f"{parse_rate(row['White Hit Rate (%)'])} & "
            f"{int(row['Black Stops']):,} & "
            f"{int(row['Black Searches']):,} & "
            f"{parse_rate(row['Black Search Rate (%)'])} & "
            f"{parse_rate(row['Black Hit Rate (%)'])} \\\\"
        )

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize"
        r" \textit{Note:} Search rates represent the proportion of stops "
        r"that resulted in a discretionary search. Hit rates represent the "
        r"proportion of discretionary searches that recovered contraband. "
        r"95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). "
        r"\textsuperscript{\dag} Agency excluded from "
        r"Figure~\ref{fig:agency_rates} due to fewer than 30 "
        r"Black/African American searches or a 95\% confidence interval "
        r"exceeding $\pm$20 percentage points for the Black hit rate estimate. "
        r"Data pooled across 2022--2024.}"
    )
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    with open("../output/tables/agency_hits.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"Table exported: agency_hits.tex "
          f"({df['Excluded'].sum()} agencies flagged with dagger)")
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



def export_enhancement_category_table_to_latex(enhancement_category):
    """
    Export enhancement rates by charge category, race, and statute level to LaTeX.
    Assumes enhancement_category already excludes 'Other' race, 'Other' charge category,
    and DUI. Only includes Assault/Violence, Weapons, and Obstruct/Resist Officer.
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

    by_cat = enhancement_category.copy()

    by_cat["Number of Cases"] = by_cat["Number of Cases"].apply(fmt_int_with_commas)
    by_cat["Enhancement Rate (%)"] = by_cat["Enhancement Rate (%)"].apply(latexify_ci_string)

    by_cat = by_cat.rename(columns={
        "Charge Category":      "Category",
        "Canonical Race":       "Race",
        "Statute Level":        "Statute Level",
        "Number of Cases":      "Cases",
        "Enhancement Rate (%)": r"Enhancement Rate (\%)",
    })

    by_cat["Category"] = by_cat["Category"].apply(
        lambda x: rf"\textbf{{{x}}}" if x != "" else x
    )

    by_cat_latex = by_cat.to_latex(
        index=False,
        escape=False,
        column_format="lllrr",
        caption="Enhancement Rates by Charge Category, Race, and Statute Level",
        label="tab:enhancement_by_category",
        na_rep="---",
    )

    # Insert separators between category blocks and race blocks
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
            next_has_row    = "&" in next_line and r"\bottomrule" not in next_line

            if current_has_row and next_has_row:
                next_parts    = [p.strip() for p in next_line.split("&")]
                next_category = next_parts[0] if len(next_parts) > 0 else ""
                next_race     = next_parts[1] if len(next_parts) > 1 else ""
                current_parts = [p.strip() for p in line.split("&")]
                current_race  = current_parts[1] if len(current_parts) > 1 else ""

                if next_category != "":
                    # New category block — full horizontal rule
                    new_lines.append(r"\addlinespace")
                    new_lines.append(r"\midrule")
                elif next_race != "" and current_race == "":
                    # New race within same category — partial rule
                    new_lines.append(r"\noalign{\vskip 2pt}")
                    new_lines.append(r"\cline{2-5}")
                    new_lines.append(r"\noalign{\vskip 2pt}")

    by_cat_latex = "\n".join(new_lines)

    note = (
        r"\smallskip" + "\n"
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} "
        r"95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). "
        r"DUI cases are excluded because the enhancement flag in those cases "
        r"reflects the Orange County DA's standard practice of jointly filing "
        r"VC~23152(a) and VC~23152(b) as paired counts rather than a traditional "
        r"sentencing enhancement: 15{,}792 of 17{,}675 DUI cases (89.3\%) had an "
        r"enhancement charge, of which 16{,}893 of 16{,}899 enhancement charges "
        r"were VC~23152(a) or 23152(b). "
        r"The ``Other'' race category and ``Other'' charge category are omitted. "
        r"Categories with near-zero enhancement rates across all racial groups or "
        r"insufficient subgroup counts for reliable inference are also omitted. "
        r"Categories are ordered by offense severity.}"
    )

    by_cat_latex = by_cat_latex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n" + note,
    )

    with open("../output/tables/enhancement_by_category.tex", "w", encoding="utf-8") as f:
        f.write(by_cat_latex)

    print("Table exported: enhancement_by_category.tex")
    return by_cat_latex



def export_wobbler_combined_to_latex(wobbler_combined):
    """
    Export the combined wobbler table (Overall + by charge category) to LaTeX.
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

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    combined = wobbler_combined.copy().reset_index()

    # Format count columns
    for col in ["Total Wobblers", "Filed as Felony", "Filed as Misdemeanor"]:
        combined[col] = combined[col].apply(fmt_int_with_commas)

    combined["Felony Rate (%) [95% CI]"] = (
        combined["Felony Rate (%) [95% CI]"].apply(latexify_ci_string)
    )

    # Blank repeated category labels so each category prints once
    combined["Charge Category"] = combined["Charge Category"].astype(object)
    combined["Charge Category"] = combined["Charge Category"].mask(
        combined["Charge Category"].duplicated(), ""
    )

    combined = combined.rename(columns={
        "Charge Category":          "Category",
        "Canonical Race":           "Race",
        "Total Wobblers":           "Total",
        "Filed as Felony":          "Felony",
        "Filed as Misdemeanor":     "Misdemeanor",
        "Felony Rate (%) [95% CI]": r"Felony Rate (\%)",
    })

    # ------------------------------------------------------------------
    # Build LaTeX manually so we can control group separators
    # ------------------------------------------------------------------
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{Felony Filing Rates for Wobbler Charges by Race and Charge Category}"
    )
    latex.append(r"\label{tab:wobbler_combined}")
    latex.append(r"\begin{tabular}{ll rrrr}")
    latex.append(r"\toprule")
    latex.append(
        r"Category & Race & Total & Felony & Misdemeanor & Felony Rate (\%) \\"
    )
    latex.append(r"\midrule")

    prev_category = None
    for _, row in combined.iterrows():
        category = row["Category"]
        race     = row["Race"]
        total    = row["Total"]
        felony   = row["Felony"]
        misdem   = row["Misdemeanor"]
        rate     = row[r"Felony Rate (\%)"]

        # Insert separator between groups (when a new non-blank category starts)
        if category != "" and prev_category is not None:
            latex.append(r"\addlinespace")
            latex.append(r"\midrule")

        # Bold the Overall category label for visual prominence
        if category == "Overall":
            cat_cell = r"\textit{Overall}"
        else:
            cat_cell = category

        latex.append(
            f"{cat_cell} & {race} & {total} & {felony} & {misdem} & {rate} \\\\"
        )

        if category != "":
            prev_category = category

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\smallskip")
    latex.append(
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} "
        r"95\% confidence intervals shown in parentheses ($\pm$1.96 SE). "
        r"The Overall row aggregates across all charge categories. "
        r"Category-specific rows are restricted to categories with more than "
        r"500 total wobbler charges and an overall felony filing rate of at "
        r"least 10\%. Categories are ordered by total wobbler volume in "
        r"descending order.}"
    )
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    with open("../output/tables/wobbler_combined.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    print("Table exported: wobbler_combined.tex")
    return latex_str



def export_statute_table_to_latex(consolidated,
                                   output_path="../output/tables/enhancement_by_statute.tex"):
    """
    Export enhancement rates by statute and race to LaTeX.
    Rates represent the proportion of all non-DUI cases for each racial
    group that contained at least one enhancement charge of that statute.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def latexify_ci_string(x):
        """Convert '0.31 (±0.18)' to LaTeX scriptsize format."""
        if pd.isna(x) or str(x).strip() in ("---", "nan"):
            return "---"
        x = str(x)
        return re.sub(r"\(±([^)]+)\)", r"({\\scriptsize $\\pm$" + r"\1})", x)

    def fmt_int_with_commas(x):
        if pd.isna(x):
            return "---"
        return f"{int(x):,}"

    tbl = consolidated.copy()
    tbl = tbl.drop(columns=["Other"], errors="ignore")
    tbl = tbl.drop(columns=["% of Enhancements"], errors="ignore")

    # Format columns
    tbl["Total"] = tbl["Total"].apply(fmt_int_with_commas)

    for col in ["Black/African American", "Hispanic/Latino",
                "White", "Asian", "Other"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(latexify_ci_string)

    # Rename columns for LaTeX
    tbl = tbl.rename(columns={
        "Statute":                "Statute",
        "Description":            "Description",
        "Total":                  "Total",
        "Black/African American": "Black/Afr. American",
        "Hispanic/Latino":        "Hispanic/Latino",
        "White":                  "White",
        "Asian":                  "Asian",
        "Other":                  "Other",
    })

    # Bold statute column
    tbl["Statute"] = tbl["Statute"].apply(lambda x: rf"\textbf{{{x}}}")

    latex_str = tbl.to_latex(
        index=False,
        escape=False,
        column_format="p{1.5cm}p{4cm}rrrr",
        caption=(
            "Enhancement Rates by Statute and Race"),
        label="tab:enhancement_by_statute",
        na_rep="---",
    )

    latex_str = latex_str.replace(
        r"\begin{tabular}",
        r"\small" + "\n" + r"\begin{tabular}",
    )

    # Add note below table
    note = (
        r"\smallskip" + "\n"
        r"\parbox{\textwidth}{\footnotesize \textit{Note:} "
        r"95\% confidence intervals shown in parentheses ($\pm 1.96$ SE). "
        r"Rates are computed as the number of non-DUI cases for each racial "
        r"group containing at least one enhancement charge of the given statute, "
        r"divided by the total number of non-DUI cases for that group. "
        r"DUI cases are excluded because the enhancement flag in those cases "
        r"reflects the DA's standard practice of filing VC~23152(a) and "
        r"VC~23152(b) as paired counts rather than a traditional sentencing "
        r"enhancement. The ``Other'' race category is omitted from the figure "
        r"but shown here for completeness. "
        r"Statutes are ordered by total enhancement count in descending order.}"
    )

    latex_str = latex_str.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n" + note,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"Table exported: {output_path}")
    return latex_str