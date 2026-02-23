"""
Stratified policing analysis functions for deeper racial disparity investigation.

These functions allow stratification by:
- Reason for contact (traffic vs. reasonable suspicion)
- Search basis type (discretionary vs. non-discretionary)
- City
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import visualization_setup


def stratify_by_reason(df, census):
    """
    Stratify policing analysis by reason for contact.
    
    Compares:
    - Traffic stops (Moving/Equipment/Non-moving violations)
    - Reasonable suspicion stops
    
    Returns separate summary tables for each type.
    """
    # Traffic stops
    traffic = df[df['reason_for_contact'].isin([
        'Moving violation',
        'Equipment violation', 
        'Non-moving violation'
    ])].copy()
    
    # Reasonable suspicion stops
    suspicion = df[df['reason_for_contact'] == 'Suspect criminal activity'].copy()
    
    def summarize(subset, stop_type):
        g = subset.groupby(['year', 'race_std'])
        
        summary = g.agg(
            Stop_Count=('action_any_search', 'size'),
            Search_Count=('action_any_search', 'sum'),
            Hit_Count=('contraband_any', 'sum'),
        ).reset_index()
        
        summary = summary.rename(columns={
            'year': 'Year',
            'race_std': 'Perceived Race',
            'Stop_Count': 'Stop Count',
            'Search_Count': 'Search Count',
            'Hit_Count': 'Hit Count',
        })
        
        summary['Search Rate'] = summary['Search Count'] / summary['Stop Count']
        summary['Hit Rate'] = np.where(
            summary['Search Count'] > 0,
            summary['Hit Count'] / summary['Search Count'],
            np.nan
        )
        
        summary['Population'] = summary['Perceived Race'].map(census)
        
        per = 1000
        summary['Stops per 1,000'] = (summary['Stop Count'] / summary['Population']) * per
        summary['Searches per 1,000'] = (summary['Search Count'] / summary['Population']) * per
        
        summary['Stop Type'] = stop_type
        
        return summary
    
    traffic_summary = summarize(traffic, 'Traffic')
    suspicion_summary = summarize(suspicion, 'Reasonable Suspicion')
    
    return traffic_summary, suspicion_summary


def stratify_by_search_basis(df, census):
    """
    Stratify policing analysis by discretionary vs. non-discretionary search basis.
    
    Discretionary: consent, plain view, plain smell, officer safety, canine
    Non-discretionary: warrant, probation/parole, incident to arrest, vehicle inventory
    
    Returns separate summary tables for each type.
    """
    # Only look at stops where a search occurred
    searches = df[df['action_any_search'] == True].copy()
    
    # Discretionary searches
    disc = searches[searches['search_type'] == 'Discretionary only'].copy()
    
    # Non-discretionary searches  
    nondisc = searches[searches['search_type'].isin([
        'Non-discretionary only',
        'Mixed discretionary and non-discretionary'
    ])].copy()
    
    def summarize(subset, basis_type):
        g = subset.groupby(['year', 'race_std'])
        
        summary = g.agg(
            Search_Count=('contraband_any', 'size'),
            Hit_Count=('contraband_any', 'sum'),
        ).reset_index()
        
        summary = summary.rename(columns={
            'year': 'Year',
            'race_std': 'Perceived Race',
            'Search_Count': 'Search Count',
            'Hit_Count': 'Hit Count',
        })
        
        summary['Hit Rate'] = summary['Hit Count'] / summary['Search Count']
        summary['Population'] = summary['Perceived Race'].map(census)
        
        per = 1000
        summary['Searches per 1,000'] = (summary['Search Count'] / summary['Population']) * per
        
        summary['Search Basis Type'] = basis_type
        
        return summary
    
    disc_summary = summarize(disc, 'Discretionary')
    nondisc_summary = summarize(nondisc, 'Non-Discretionary')
    
    return disc_summary, nondisc_summary


def stratify_by_city(df, census, top_n=10):
    """
    Stratify policing analysis by city.
    
    Parameters:
    -----------
    df : DataFrame
        Policing data
    census : Series
        Population by race
    top_n : int
        Number of top cities to analyze
        
    Returns:
    --------
    dict of DataFrames, one per city
    """
    # Get top N cities by stop count
    top_cities = df['closest_city'].value_counts().head(top_n).index.tolist()
    
    city_summaries = {}
    
    for city in top_cities:
        city_data = df[df['closest_city'] == city].copy()
        
        g = city_data.groupby(['year', 'race_std'])
        
        summary = g.agg(
            Stop_Count=('action_any_search', 'size'),
            Search_Count=('action_any_search', 'sum'),
            Hit_Count=('contraband_any', 'sum'),
        ).reset_index()
        
        summary = summary.rename(columns={
            'year': 'Year',
            'race_std': 'Perceived Race',
            'Stop_Count': 'Stop Count',
            'Search_Count': 'Search Count',
            'Hit_Count': 'Hit Count',
        })
        
        summary['Search Rate'] = summary['Search Count'] / summary['Stop Count']
        summary['Hit Rate'] = np.where(
            summary['Search Count'] > 0,
            summary['Hit Count'] / summary['Search Count'],
            np.nan
        )
        
        # Note: Using county-level census as city-level not available
        summary['Population'] = summary['Perceived Race'].map(census)
        
        summary['City'] = city
        
        city_summaries[city] = summary
    
    return city_summaries


def compare_strata(strata_dict, metric, title_prefix=""):
    """
    Create comparison plots across strata.
    
    Parameters:
    -----------
    strata_dict : dict
        Dictionary mapping stratum name to DataFrame
    metric : str
        Column name to plot (e.g., 'Search Rate', 'Hit Rate')
    title_prefix : str
        Prefix for plot title
    """
    race_col = 'Perceived Race'
    
    # Get all unique races and years across all strata
    all_races = set()
    all_years = set()
    
    for df in strata_dict.values():
        if race_col in df.columns:
            all_races.update(df[race_col].unique())
        if 'Year' in df.columns:
            all_years.update(df['Year'].unique())
    
    races = sorted(all_races)
    years = sorted(all_years)
    
    # Create subplots for each race
    n_races = len(races)
    fig, axes = plt.subplots(n_races, 1, figsize=(10, 3*n_races), sharex=True)
    
    if n_races == 1:
        axes = [axes]
    
    for idx, race in enumerate(races):
        ax = axes[idx]
        
        for stratum_name, df in strata_dict.items():
            race_data = df[df[race_col] == race].sort_values('Year')
            
            if len(race_data) > 0 and metric in race_data.columns:
                ax.plot(race_data['Year'], race_data[metric], 
                       marker='o', label=stratum_name)
        
        ax.set_ylabel(metric)
        ax.set_title(f"{race}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Year')
    fig.suptitle(f"{title_prefix}{metric} by Race and Stratum", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


def summarize_disparities_by_stratum(strata_dict, reference_race='White'):
    """
    Calculate disparity ratios (compared to reference race) for each stratum.
    
    Uses most recent year's data.
    
    Returns DataFrame with columns:
    - Stratum
    - Perceived Race  
    - Search Rate Ratio
    - Hit Rate Ratio
    - Stops per 1,000 Ratio
    """
    results = []
    
    for stratum_name, df in strata_dict.items():
        # Get most recent year
        max_year = df['Year'].max()
        recent = df[df['Year'] == max_year].copy()
        
        # Get reference race values
        ref = recent[recent['Perceived Race'] == reference_race]
        
        if len(ref) == 0:
            continue
            
        ref_search_rate = ref['Search Rate'].iloc[0] if 'Search Rate' in ref.columns else np.nan
        ref_hit_rate = ref['Hit Rate'].iloc[0] if 'Hit Rate' in ref.columns else np.nan
        ref_stops_per_k = ref['Stops per 1,000'].iloc[0] if 'Stops per 1,000' in recent.columns else np.nan
        
        for _, row in recent.iterrows():
            race = row['Perceived Race']
            
            result = {
                'Stratum': stratum_name,
                'Perceived Race': race,
                'Year': max_year
            }
            
            if 'Search Rate' in recent.columns and ref_search_rate > 0:
                result['Search Rate Ratio'] = row['Search Rate'] / ref_search_rate
                
            if 'Hit Rate' in recent.columns and ref_hit_rate > 0:
                result['Hit Rate Ratio'] = row['Hit Rate'] / ref_hit_rate
                
            if 'Stops per 1,000' in recent.columns and ref_stops_per_k > 0:
                result['Stops per 1,000 Ratio'] = row['Stops per 1,000'] / ref_stops_per_k
            
            results.append(result)
    
    return pd.DataFrame(results)


def visualize_reason_stratification(traffic_summary, suspicion_summary):
    """
    Create visualizations comparing traffic vs. suspicion stops.
    """
    strata = {
        'Traffic Stops': traffic_summary,
        'Reasonable Suspicion': suspicion_summary
    }
    
    print("=" * 60)
    print("STRATIFICATION BY REASON FOR CONTACT")
    print("=" * 60)
    
    compare_strata(strata, 'Search Rate', 
                  title_prefix="Stop Reason Comparison: ")
    
    compare_strata(strata, 'Hit Rate',
                  title_prefix="Stop Reason Comparison: ")
    
    # Show disparity summary
    disparities = summarize_disparities_by_stratum(strata)
    print("\n=== Disparity Ratios (Relative to White) ===")
    print(disparities.to_string(index=False))
    
    return disparities


def visualize_search_basis_stratification(disc_summary, nondisc_summary):
    """
    Create visualizations comparing discretionary vs. non-discretionary searches.
    """
    strata = {
        'Discretionary': disc_summary,
        'Non-Discretionary': nondisc_summary
    }
    
    print("=" * 60)
    print("STRATIFICATION BY SEARCH BASIS")
    print("=" * 60)
    
    compare_strata(strata, 'Hit Rate',
                  title_prefix="Search Basis Comparison: ")
    
    compare_strata(strata, 'Searches per 1,000',
                  title_prefix="Search Basis Comparison: ")
    
    disparities = summarize_disparities_by_stratum(strata)
    print("\n=== Disparity Ratios (Relative to White) ===")
    print(disparities.to_string(index=False))
    
    return disparities


def visualize_city_stratification(city_summaries, metric='Search Rate'):
    """
    Create visualizations comparing across cities.
    
    Shows top 5 cities with highest disparities.
    """
    print("=" * 60)
    print("STRATIFICATION BY CITY")
    print("=" * 60)
    
    # Show first 5 cities
    top_5 = dict(list(city_summaries.items())[:5])
    
    compare_strata(top_5, metric,
                  title_prefix=f"City Comparison: ")
    
    # Calculate disparities for all cities
    disparities = summarize_disparities_by_stratum(city_summaries)
    
    print(f"\n=== Top Cities by {metric} Disparity (Black vs White) ===")
    black_disparities = disparities[
        (disparities['Perceived Race'] == 'Black/African American') &
        (disparities[f'{metric} Ratio'].notna())
    ].sort_values(f'{metric} Ratio', ascending=False)
    
    print(black_disparities[['Stratum', f'{metric} Ratio']].head(10).to_string(index=False))
    
    return disparities
