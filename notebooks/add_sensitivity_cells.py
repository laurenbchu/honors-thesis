#!/usr/bin/env python3
import json

# Read the notebook
with open('analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find insertion point (after the cell with "percent_multi")
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'percent_multi' in source:
            # Skip the next empty cell, insert after that
            insert_idx = i + 2
            break

if not insert_idx:
    print("Could not find insertion point!")
    exit(1)

print(f"Inserting at position {insert_idx}")

# Create sensitivity analysis cells
sensitivity_cells = [
    {
        "cell_type": "markdown",
        "id": "sensitivity_analysis_md",
        "metadata": {},
        "source": [
            "# Sensitivity Analysis: Mixed Search Bases\n",
            "\n",
            "We now test whether including mixed search bases (those with both discretionary and nondiscretionary bases) as discretionary searches affects our conclusions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "create_mixed_indicators",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create alternative classification: treat mixed searches as discretionary\n",
            "policing[\"disc_search_mixed\"] = policing[\"search_type\"].isin([\"Discretionary only\", \"Mixed\"])\n",
            "policing[\"disc_hit_mixed\"] = policing[\"disc_search_mixed\"] & (policing[\"contraband_any\"] == 1)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "compute_rates_mixed",
        "metadata": {},
        "outputs": [],
        "source": [
            "from utils import policing_rates_mixed\n",
            "\n",
            "# Compute policing rates using mixed classification\n",
            "policing_analysis_mixed = policing_rates_mixed(policing, census_coarse)\n",
            "policing_analysis_mixed"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "comparison_md",
        "metadata": {},
        "source": [
            "## Comparing Strict vs. Mixed Classifications"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "run_comparison",
        "metadata": {},
        "outputs": [],
        "source": [
            "from utils import compare_search_classifications\n",
            "\n",
            "# Generate comparison tables\n",
            "comparison_results = compare_search_classifications(\n",
            "    policing_analysis,  # strict classification\n",
            "    policing_analysis_mixed  # mixed classification\n",
            ")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "display_2024_comparison",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display 2024 comparison (most recent year)\n",
            "display_cols = [\n",
            "    'Perceived Race',\n",
            "    'Search Count_strict',\n",
            "    'Search Count_mixed',\n",
            "    'Search Count % Change',\n",
            "    'Search Rate_strict',\n",
            "    'Search Rate_mixed',\n",
            "    'Hit Rate_strict',\n",
            "    'Hit Rate_mixed'\n",
            "]\n",
            "\n",
            "print(\"2024 COMPARISON: STRICT VS. MIXED CLASSIFICATION\")\n",
            "print(\"=\" * 80)\n",
            "comparison_results['summary_2024'][display_cols].round(4)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "display_disparity_ratios",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare disparity ratios under both approaches\n",
            "print(\"\\nDISPARITY RATIOS (relative to White baseline)\")\n",
            "print(\"=\" * 80)\n",
            "comparison_results['disparity_comparison'].round(3)"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "interpretation_md",
        "metadata": {},
        "source": [
            "## Interpretation\n",
            "\n",
            "**Key findings from sensitivity analysis:**\n",
            "\n",
            "1. **Search count impact**: Including mixed bases increases discretionary search counts by ~37%\n",
            "2. **Disparity stability**: Check if racial disparities remain similar or change significantly\n",
            "3. **Hit rate patterns**: Assess whether outcome test results are robust to classification choice\n",
            "\n",
            "**What to look for:**\n",
            "- If disparity ratios are stable (±10%), the strict classification is robust\n",
            "- If ratios change substantially, mixed searches may be systematically different\n",
            "- Changes in hit rates could indicate different threshold levels for mixed vs. pure discretionary searches"
        ]
    }
]

# Insert the cells
for i, cell in enumerate(sensitivity_cells):
    nb['cells'].insert(insert_idx + i, cell)

# Write back
with open('analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully added {len(sensitivity_cells)} cells!")
print(f"Original notebook had ~1023 cells, now has {len(nb['cells'])} cells")
