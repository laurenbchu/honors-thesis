# Data Science Honors Thesis: Racial Policing and Prosecution Disparities
Repository for Lauren Chu's Data Science Honors Thesis, Class of Spring 2026 at UC Berkeley

[![Binder](https://mybinder.org/badge_logo.svg)]([https://mybinder.org/v2/gh/ls88-openscienceconnector/final-project/master](https://mybinder.org/v2/gh/laurenbchu/honors-thesis/main))

## Contributors
_Lauren Chu_

## Overview
I aim to answer the question: _How do geographical disparities in policing relate to racial disparities in prosecution?_ While there exist many papers covering racial disparities in policing and racial disparities in prosecution, I hope to identify the relationship between the two stages within the criminal justice system so as to better inform criminal justice reform policy. I’ll focus on Orange County as a case study, with possible extensions to San Francisco and Contra Costa Counties, depending on data availability from the ACLU.

## Advisor
I am advised by Professor Joshua Grossman.

## Dataset
I will primarily utilize the Racial and Identity Profiling Act (RIPA) data to identify geographical disparities in policing, which is a California-wide dataset with 14 tables on stops, searches, and seizures by hour, agency, month, location, primary reason for stop, citation, demographic information, and more. 	Because there is so much data and it’s not fully cleaned, I will write a data cleaning pipeline to aid in my analysis. 

I will also utilize the ACLU’s Racial Justice Act (RJA) data, which has annual information per county on cases filed, charges filed, sentences, defendant information, etc. Professor Grossman has been in contact with Jason Tsui from the Center for Policing Equity (CPE), who has already begun to process the data for Orange County; as such, I plan to analyze Orange County on both the policing and prosecution level for easy comparison. Finally, I will supplement my findings with demographic information from the US Census to act as benchmarks for population demographics in the county and cities I’m analyzing.

## Data Techniques
To conduct my EDA, I will first begin by checking for missing data points and duplicates within the RIPA data, as well as how to join together multiple tables to gain the most information per record. In order to only choose Orange County data, as well as analyze on a more granular level, I think it would be useful to add a geocode to each record for easier geographical analysis. For the RJA data, I will make sure that all of the categories per charge remain consistent over the years that I analyze, as well as identify the county or city averages for different prosecution outcomes (e.g. dismissal, plea, trial, sentence length). Using the Census data, I’ll identify the demographics for each county / city that I analyze. 

Some ideas I have for evaluating my central question is to conduct a cost benefit analysis, in which I compare the rates of different policing methods with crime rates across cities or counties to see if higher police activity is correlated with reductions in crimes, because if not, then the disparate racial impact of the policing methods cannot be justified. I can also build a statistical model to estimate the likelihood of a stop resulting in a certain charge, which I can then use as a control for “risk” when comparing treatment across racial groups for more accurate analysis. To connect both of my datasets, I can identify stop, search, and “hit” rates to identify disparity ratios by racial group, and then conduct a similar process with prosecution filings, convictions, and sentencing. Once I have the concrete measures for both stages, I can conduct a correlation analysis and run a regression model if I have time, controlling for demographics. 
