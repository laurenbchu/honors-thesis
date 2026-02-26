# Sensitivity Analysis: Mixed Search Bases

## Overview

This sensitivity analysis tests whether including searches with **mixed bases** (both discretionary and nondiscretionary) as discretionary searches affects the conclusions about racial disparities.

## Background

Currently, 4.79% of search cases (15,406 stops) have multiple search bases recorded, containing a mix of:
- **Discretionary bases**: consent, plain view, plain smell, canine
- **Nondiscretionary bases**: warrant, probation, incident to arrest, vehicle inventory, etc.

The main analysis uses a **strict classification** that only counts "Discretionary only" searches (41,137 stops).

## What Was Added

### New Functions in `utils.py`:

1. **`policing_rates_mixed(df, census)`**
   - Alternative to `policing_rates()` 
   - Treats mixed search bases as discretionary
   - Uses columns: `disc_search_mixed` and `disc_hit_mixed`

2. **`compare_search_classifications(policing_strict, policing_mixed)`**
   - Generates comparison tables between the two approaches
   - Returns dictionary with:
     - `'by_year_race'`: Full year-by-year comparison
     - `'summary_2024'`: Most recent year summary
     - `'disparity_comparison'`: Disparity ratios under both approaches

### New Cells in `analysis.ipynb`:

Seven new cells added after the "Handling Multiple Search Bases" section:

1. **Markdown header**: Explains sensitivity analysis purpose
2. **Create indicators**: Generates `disc_search_mixed` and `disc_hit_mixed` columns
3. **Compute rates**: Runs `policing_rates_mixed()` function
4. **Comparison header**: Section divider
5. **Run comparison**: Calls `compare_search_classifications()`
6. **Display 2024 results**: Shows search counts and rates for both classifications
7. **Display disparities**: Shows how disparity ratios change

## How to Use

1. **Run the notebook** through the new sensitivity analysis section
2. **Examine the comparison tables** to see:
   - How much search counts increase (~37% expected)
   - Whether search rates change substantially by race
   - Whether disparity ratios remain stable
   - Whether hit rates change (indicating different threshold levels)

## Interpretation Guidelines

### Robust Results (Good News)
- Disparity ratios change by <10%
- Hit rate patterns remain consistent
- Conclusions about racial disparities hold under both classifications

### Sensitivity Concerns (Need Discussion)
- Disparity ratios change by >20%
- Hit rates systematically different for mixed searches
- Mixed searches disproportionately affect certain racial groups

### What to Report in Thesis

If results are **robust**:
- Report strict classification as primary results
- Note in a footnote or appendix: "Results robust to including mixed search bases"
- Show key comparison statistics in appendix

If results are **sensitive**:
- Report both classifications
- Discuss why mixed searches might differ
- Consider which classification better captures discretionary police behavior
- May need additional analysis of what makes a search "mixed"

## Next Steps

After running the analysis:

1. **Document findings** in your thesis
2. **Consider**: Do mixed searches represent:
   - Officers using multiple legal justifications?
   - Data quality issues?
   - A continuum of discretion?
3. **Discuss with advisor** which classification to use as primary analysis

## Technical Notes

- Mixed classification increases N by 37% (from 41,137 to 56,543)
- All per-capita calculations use 2020 Census population data
- Disparity ratios calculated relative to White baseline
- Functions handle missing values (NaN) appropriately in hit rate calculations
