def create_latex_table(data, category_name):
    """
    Create a publishable LaTeX table for a specific charge category.
    Matches the format requested with races grouped by statute level.
    """
    # Filter data for this category
    cat_data = data[data['charge_category'] == category_name].copy()
    
    # Get unique years and statute levels
    years = sorted(cat_data['Year'].unique())
    
    # Define race ordering (most represented to least)
    race_order = ['Black/African American', 'Hispanic/Latino', 'White', 'Asian', 'Other']
    races_present = [r for r in race_order if r in cat_data['race_std'].values]
    
    # Start building the LaTeX table
    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    
    # Create a clean table name
    table_label = category_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    
    latex.append(f"\\caption{{Prosecution for {category_name}: Enhancement Rates by Race, Statute Level, and Year (\\%)}}")
    latex.append(f"\\label{{tab:prosecution_{table_label}}}")
    latex.append("\\begin{tabular}{l l " + "c " * len(years) + "}")
    latex.append("\\toprule")
    
    # Create header row
    header = ["Race", "Statute Level"] + [str(year) for year in years]
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\midrule")
    latex.append("")
    
    # Add data rows grouped by race
    for i, race in enumerate(races_present):
        race_data = cat_data[cat_data['race_std'] == race]
        
        # Add rows for Felony and Misdemeanor
        for j, level in enumerate(['Felony', 'Misdemeanor']):
            level_data = race_data[race_data['statute_level'] == level]
            
            # First column: race name (only on first row for this race)
            if j == 0:
                row = [race]
            else:
                row = [""]
            
            # Second column: statute level
            row.append(level)
            
            # Add enhancement rates for each year with confidence intervals
            for year in years:
                year_data = level_data[level_data['Year'] == year]
                if len(year_data) > 0:
                    rate = year_data['Enhancement Rate'].iloc[0] * 100
                    se = year_data['SE'].iloc[0] * 100
                    ci = 1.96 * se
                    row.append(f"{rate:.2f} ({{\\scriptsize ±{ci:.2f}}})")
                else:
                    row.append("—")
            
            latex.append(" & ".join(row) + " \\\\")
        
        # Add space between races (except after the last one)
        if i < len(races_present) - 1:
            latex.append("\\addlinespace")
        latex.append("")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\smallskip")
    latex.append("\\parbox{\\textwidth}{\\footnotesize \\textit{Note:} 95\\% confidence intervals shown in parentheses (±1.96 SE).}")
    latex.append("\\end{table}")
    latex.append("")
    
    return "\n".join(latex)

# Create output directory if it doesn't exist
import os
output_dir = "../output/latex_tables"
os.makedirs(output_dir, exist_ok=True)

# Generate tables for each category
print("\nGenerating LaTeX tables...")
for category in top_categories:
    print(f"  - {category}")
    latex_table = create_latex_table(enhancement_top6, category)
    
    # Save to file
    filename = f"{category.lower().replace(' ', '_').replace('/', '_')}_enhancement_table.tex"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(latex_table)
    
    print(f"    Saved to: {filepath}")

# Also create a combined file
combined_filepath = os.path.join(output_dir, "all_enhancement_tables.tex")
with open(combined_filepath, 'w') as f:
    f.write("% Enhancement Rate Tables by Charge Category\n")
    f.write("% Generated from analysis.ipynb\n\n")
    
    for category in top_categories:
        latex_table = create_latex_table(enhancement_top6, category)
        f.write(latex_table)
        f.write("\n\\clearpage\n\n")

print(f"\nCombined file saved to: {combined_filepath}")
print("\nDone!")
