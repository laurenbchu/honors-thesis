# Orange County Policing and Prosecution Analysis
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/laurenbchu/honors-thesis/main)

This repository was created by Lauren Chu as part of a UC Berkeley Data Science honors thesis advised by Joshua Grossman. It contains the code, processed inputs, and exported outputs for my undergraduate honors thesis on racial disparities in policing and prosecution in Orange County, California.

The project compares discretionary decision-making across two stages of the criminal legal system:

- **Policing** using Orange County RIPA stop data from 2022–2024
- **Prosecution** using cleaned Orange County prosecution data from 2021–2023

The analysis combines these sources with 2020 U.S. Census population denominators to produce the figures and LaTeX tables used in the thesis.

## Repository Structure

```text
.
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

## Main Components

- `notebooks/01_ripa_cleaning.ipynb` downloads, harmonizes, and cleans Orange County RIPA stop data.
- `notebooks/02_analysis.ipynb` runs the policing and prosecution analyses used in the thesis.
- `data/processed/` stores processed input files used by the analysis.
- `data/metadata/` stores source documentation and data dictionaries.
- `output/` contains thesis-ready figures and tables.

## Data

This repository includes processed input data needed for replication, including the cleaned prosecution dataset. The policing notebook downloads the original RIPA source files directly from the California DOJ/OpenJustice portal.

Supporting source documentation is stored in `data/metadata/`.

## Reproducibility

Detailed setup instructions, execution order, dependencies, and reproducibility notes are provided in [`docs/reproducibility.md`](docs/reproducibility.md).
