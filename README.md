# Orange County Policing and Prosecution Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/laurenbchu/honors-thesis/main)

This repository contains the code, processed data inputs, and exported thesis outputs for my analysis of racial disparities in policing and prosecution in Orange County, California.

The project has two main stages:

1. **RIPA Cleaning**: Download and harmonize Orange County RIPA stop data for 2022–2024 and save a cleaned policing file.
2. **Analysis**: Combine the cleaned policing data, cleaned prosecution data, and 2020 Census population denominators to generate the figures and LaTeX tables used in the thesis.

## Repository Structure

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   └── cleaned_orange_aclu_2021_2023.csv
│   └── metadata/
│       ├── RIPA_2022_ReadMe.pdf
│       ├── RIPA_2023_ReadMe.pdf
│       ├── RIPA_2024_ReadMe.pdf
│       ├── RJA_database_columns_documentation.xlsx
│       └── data-source-documentation.md
├── docs/
│   └── reproducibility.md
├── notebooks/
│   ├── 01_ripa_cleaning.ipynb
│   └── 02_analysis.ipynb
├── output/
│   ├── figures/
│   └── tables/
└── src/
    ├── __init__.py
    ├── analysis_utils.py
    ├── table_utils.py
    └── visualization_utils.py
```

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/laurenbchu/honors-thesis.git
cd honors-thesis
pip install -r requirements.txt
```

Run the notebooks in this order:

1. `notebooks/01_ripa_cleaning.ipynb`
2. `notebooks/02_analysis.ipynb`

For full setup notes and reproduction details, see [`docs/reproducibility.md`](docs/reproducibility.md).

## Project Workflow

### 1. `notebooks/01_ripa_cleaning.ipynb`

This notebook downloads Orange County RIPA stop data for 2022, 2023, and 2024 from the California DOJ/OpenJustice portal, harmonizes the 2024 schema to match earlier years, creates derived analysis variables, and exports a cleaned policing dataset to:

```text
data/processed/cleaned_orange_ripa_2022_2024.csv
```

### 2. `notebooks/02_analysis.ipynb`

This notebook loads:

- the cleaned RIPA policing file generated in step 1
- the cleaned Orange County prosecution dataset for 2021–2023
- 2020 Decennial Census PL Table P2 population data from the U.S. Census API

It then produces the full thesis analysis, including:

- policing stop, search, and hit rate analyses
- policing analyses by reason for contact
- agency-level Black–White contraband hit-rate comparisons
- sensitivity analyses for alternative discretionary-search definitions
- prosecution enhancement analyses
- wobbler felony-filing analyses

## Data Sources

### Policing Data

California DOJ RIPA stop data for Orange County, 2022–2024.

### Prosecution Data

Cleaned Orange County prosecution data derived from 2021–2023 RJA records:

```text
data/processed/cleaned_orange_aclu_2021_2023.csv
```

### Population Data

2020 Decennial Census PL Table P2 population data are retrieved at runtime from the U.S. Census API.

### Source Documentation

Supporting metadata and documentation are stored in:

```text
data/metadata/
```

## Outputs

The repository includes thesis-ready outputs in:

- `output/figures/` for PDF figures
- `output/tables/` for LaTeX tables

Expected figure files include:

- `policing.pdf`
- `policing_by_reason.pdf`
- `agency_hits.pdf`
- `sensitivity.pdf`
- `enhancement_rate_by_race_statute.pdf`
- `enhancement_assault_violence_weapons.pdf`
- `enhancement_dui.pdf`
- `wobblers.pdf`

Expected table files include:

- `stops_searches_per_capita.tex`
- `searches_hits.tex`
- `reason_for_contact.tex`
- `agency_hits.tex`
- `sensitivity.tex`
- `enhancement_overall.tex`
- `enhancement_by_category.tex`
- `wobbler_overall.tex`
- `wobbler_by_category.tex`

## Reproducing Exported Outputs

The analysis notebook contains the helper calls used to export figures and LaTeX tables. In the current version of the notebook, these export lines are left in place but commented out. The committed files in `output/` are the final thesis-ready outputs.

To regenerate exported files, uncomment the relevant export lines in `notebooks/02_analysis.ipynb` and rerun the notebook.

## Software

This project was run in Python using Jupyter notebooks.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The current environment is based on:

- pandas
- numpy
- matplotlib
- requests
- openpyxl
- jupyter
- adjustText
