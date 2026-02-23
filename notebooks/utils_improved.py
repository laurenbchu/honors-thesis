import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_policing(policing_analysis):
    """
    Enhanced policing visualizations with improved styling:
      1) Stops per 1,000 by Perceived Race (exposure)
      2) Searches per 1,000 by Perceived Race (exposure)
      3) Search Rate by Perceived Race (decision)
      4) Hit Rate by Perceived Race (outcome test)
    
    Uses colorblind-friendly palette, larger markers, and clearer labels.
    """
    df = policing_analysis.copy()
    race_col = "Perceived Race"

    # Use existing visualization_setup if available
    if pd.api.types.is_categorical_dtype(df[race_col]):
        races = list(df[race_col].cat.categories)
    else:
        race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
        df[race_col] = pd.Categorical(df[race_col], categories=race_order, ordered=True)
        df = df.sort_values(["Year", race_col])
        races = list(df[race_col].cat.categories)

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in policing_analysis['Year'].")

    # Colorblind-friendly palette (Wong 2011)
    colors = {
        "Black/African American": "#E69F00",  # Orange
        "Hispanic/Latino": "#56B4E9",         # Sky Blue
        "White": "#009E73",                   # Bluish Green
        "Asian": "#F0E442",                   # Yellow
        "Other": "#CC79A7"                    # Reddish Purple
    }
    
    # Distinct marker styles
    markers = {
        "Black/African American": "o",
        "Hispanic/Latino": "s",
        "White": "^",
        "Asian": "D",
        "Other": "v"
    }

    def _set_styling(ax):
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _line_chart(y_col, title, ylabel, caption):
        if y_col not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for race in races:
            d = df[df[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                ax.plot(
                    d["Year"], 
                    d[y_col], 
                    marker=markers.get(race, "o"),
                    color=colors.get(race, "#000000"),
                    linewidth=2.5,
                    markersize=9,
                    label=race,
                    alpha=0.85,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        legend = ax.legend(
            title="Perceived Race",
            title_fontsize=11,
            fontsize=10,
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95
        )
        legend.get_title().set_fontweight('bold')
        
        _set_styling(ax)
        
        # Add caption below
        fig.text(0.5, -0.01, caption, ha="center", fontsize=10, 
                style='italic', wrap=True)
        
        plt.tight_layout()
        plt.show()

    # --- Core policing figures ---
    _line_chart(
        "Stops per 1,000",
        title=f"Police Stops per 1,000 Residents by Race ({years[0]}–{years[-1]})",
        ylabel="Stops per 1,000 Residents",
        caption="Population-normalized exposure to police stops. Higher values indicate disproportionate contact with law enforcement."
    )

    _line_chart(
        "Searches per 1,000",
        title=f"Police Searches per 1,000 Residents by Race ({years[0]}–{years[-1]})",
        ylabel="Searches per 1,000 Residents",
        caption="Population-normalized exposure to searches. Combines both exposure (stops) and decision-making (search rate)."
    )

    _line_chart(
        "Search Rate",
        title=f"Search Rate by Race ({years[0]}–{years[-1]})",
        ylabel="Proportion of Stops Resulting in Search",
        caption="Conditional on being stopped: the likelihood of being searched. Isolates officer discretionary decision-making."
    )

    _line_chart(
        "Hit Rate",
        title=f"Contraband Hit Rate by Race ({years[0]}–{years[-1]})",
        ylabel="Proportion of Searches Finding Contraband",
        caption="Outcome test: success rate of searches. Lower hit rates may indicate lower evidentiary thresholds for certain groups."
    )


def visualize_prosecution(prosecution_analysis):
    """
    Enhanced prosecution visualizations:
      For each statute level (Felony/Misdemeanor):
        - Enhancement Rate by Race over time
    
    Uses colorblind-friendly palette and improved styling.
    """
    df = prosecution_analysis.copy()
    race_col = "Canonical Race"

    # Setup race ordering
    if pd.api.types.is_categorical_dtype(df[race_col]):
        races = list(df[race_col].cat.categories)
    else:
        race_order = ["Black/African American", "Hispanic/Latino", "White", "Asian", "Other"]
        df[race_col] = pd.Categorical(df[race_col], categories=race_order, ordered=True)
        df = df.sort_values(["Year", race_col])
        races = list(df[race_col].cat.categories)

    years = sorted(df["Year"].dropna().unique().tolist())
    if not years:
        raise ValueError("No years found in prosecution_analysis['Year'].")

    statute_levels = sorted(df["statute_level"].dropna().unique().tolist())
    if not statute_levels:
        raise ValueError("No statute_level values found.")

    # Colorblind-friendly palette
    colors = {
        "Black/African American": "#E69F00",
        "Hispanic/Latino": "#56B4E9",
        "White": "#009E73",
        "Asian": "#F0E442",
        "Other": "#CC79A7"
    }
    
    markers = {
        "Black/African American": "o",
        "Hispanic/Latino": "s",
        "White": "^",
        "Asian": "D",
        "Other": "v"
    }

    def _set_styling(ax):
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _line_chart(dsub, y_col, title, ylabel, caption):
        if y_col not in dsub.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for race in races:
            d = dsub[dsub[race_col] == race].sort_values("Year")
            if len(d) > 0 and d[y_col].notna().any():
                ax.plot(
                    d["Year"], 
                    d[y_col], 
                    marker=markers.get(race, "o"),
                    color=colors.get(race, "#000000"),
                    linewidth=2.5,
                    markersize=9,
                    label=race,
                    alpha=0.85,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Year", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        legend = ax.legend(
            title="Race",
            title_fontsize=11,
            fontsize=10,
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95
        )
        legend.get_title().set_fontweight('bold')
        
        _set_styling(ax)
        
        # Add caption
        fig.text(0.5, -0.01, caption, ha="center", fontsize=10, 
                style='italic', wrap=True)
        
        plt.tight_layout()
        plt.show()

    for lvl in ["Felony", "Misdemeanor"]:
        dsub = df[df["statute_level"] == lvl].copy()
        if dsub.empty:
            continue

        _line_chart(
            dsub,
            "Enhancement Rate",
            title=f"Enhancement Rate by Race: {lvl} Charges ({years[0]}–{years[-1]})",
            ylabel="Proportion of Charges with Enhancements",
            caption=f"Among {lvl.lower()} charges: the rate at which sentence enhancements are applied. Higher rates indicate harsher charging decisions."
        )


def create_disparity_comparison(policing_analysis, metric_col, baseline_race="White"):
    """
    Create a bar chart comparing latest-year disparity ratios.
    
    Parameters:
    -----------
    policing_analysis : pd.DataFrame
        Analysis table with Year, Perceived Race, and metric columns
    metric_col : str
        Column name to analyze (e.g., "Search Rate", "Hit Rate")
    baseline_race : str
        Reference group for disparity ratios
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    df = policing_analysis.copy()
    
    # Get latest year
    latest_year = df["Year"].max()
    latest = df[df["Year"] == latest_year].copy()
    
    # Calculate disparity ratios
    baseline_value = latest[latest["Perceived Race"] == baseline_race][metric_col].iloc[0]
    latest["Disparity Ratio"] = latest[metric_col] / baseline_value
    
    # Sort by disparity
    latest = latest.sort_values("Disparity Ratio", ascending=False)
    
    # Colors
    colors_map = {
        "Black/African American": "#E69F00",
        "Hispanic/Latino": "#56B4E9",
        "White": "#009E73",
        "Asian": "#F0E442",
        "Other": "#CC79A7"
    }
    
    bar_colors = [colors_map.get(r, "#999999") for r in latest["Perceived Race"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(latest["Perceived Race"], latest["Disparity Ratio"], color=bar_colors, alpha=0.8)
    
    # Add reference line at 1.0
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'{baseline_race} (baseline)')
    
    # Add value labels
    for i, (race, ratio) in enumerate(zip(latest["Perceived Race"], latest["Disparity Ratio"])):
        ax.text(ratio + 0.05, i, f'{ratio:.2f}', va='center', fontweight='bold')
    
    ax.set_xlabel("Disparity Ratio", fontsize=12, fontweight='bold')
    ax.set_ylabel("Perceived Race", fontsize=12, fontweight='bold')
    ax.set_title(f"{metric_col} Disparity Ratios ({latest_year})", fontsize=14, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    
    plt.tight_layout()
    return fig
