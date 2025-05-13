# COVID-19 Global Data Tracker
# This notebook analyzes global COVID-19 trends including cases, deaths, and vaccinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

print("COVID-19 Global Data Tracker")
print("=" * 50)

# 1. Data Collection
print("\n1. Data Collection")
print("-" * 50)

file_name = 'owid-covid-data.csv'
file_path = file_name

# Check if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Upload in Colab if file is not present
if IN_COLAB and not os.path.exists(file_path):
    from google.colab import files
    print(f"{file_name} not found. Please upload the file.")
    uploaded = files.upload()  # Manually upload the file
    if file_name not in uploaded:
        print(f"Upload failed. Please make sure the file is named '{file_name}'.")
        import sys; sys.exit(1)

# For local Jupyter, check if file exists
if not os.path.exists(file_path):
    print(f"Error: '{file_name}' not found in current directory: {os.getcwd()}")
    print("Please upload it using the Jupyter interface or move it into this folder.")
    import sys; sys.exit(1)

# Load the dataset
df = pd.read_csv(file_path)
print(f"Data loaded successfully! Shape: {df.shape}")

# 2. Data Exploration
print("\n2. Data Exploration")
print("-" * 50)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

print("\nBasic statistics for numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe().round(2).T)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in key columns:")
key_cols = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
            'people_vaccinated', 'people_fully_vaccinated']
print(missing_values[key_cols].sort_values(ascending=False))

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
print("\nDate range in the dataset:")
print(f"Start date: {df['date'].min().strftime('%Y-%m-%d')}")
print(f"End date: {df['date'].max().strftime('%Y-%m-%d')}")

# Get unique countries/locations
print(f"\nTotal number of countries/locations: {df['location'].nunique()}")
print(f"Sample of countries: {', '.join(df['location'].unique()[:10])}")

# 3. Data Cleaning
print("\n3. Data Cleaning")
print("-" * 50)

countries_of_interest = ['United States', 'India', 'Brazil', 'Kenya', 'United Kingdom', 'Germany']
df_countries = df[df['location'].isin(countries_of_interest)].copy()
print(f"Selected {len(countries_of_interest)} countries for detailed analysis")

found_countries = df_countries['location'].unique()
print(f"Found data for: {', '.join(found_countries)}")

missing_countries = set(countries_of_interest) - set(found_countries)
if missing_countries:
    print(f"Missing data for: {', '.join(missing_countries)}")

key_metrics = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
               'people_vaccinated', 'people_fully_vaccinated']
for metric in key_metrics:
    if metric in df_countries.columns: # Check against df_countries
        df_countries[metric] = df_countries.groupby('location')[metric].transform(lambda x: x.ffill())

print("Missing values handled for key metrics")

# Calculate latest data snapshot for each country (used in multiple sections)
# This DataFrame will be named _LATEST_DF to denote its special role.
_LATEST_DF = pd.DataFrame() # Initialize as empty
if not df_countries.empty and 'date' in df_countries.columns:
    try:
        _LATEST_DF = df_countries.loc[df_countries.groupby('location')['date'].idxmax()].copy()
        if _LATEST_DF.empty:
            print("\nWarning: Calculated _LATEST_DF is empty (e.g., no valid dates per group).")
        else:
            print(f"\nLatest data snapshot (_LATEST_DF) calculated for {len(_LATEST_DF)} locations.")
    except Exception as e:
        print(f"Error calculating _LATEST_DF: {e}. _LATEST_DF will be empty.")
else:
    if df_countries.empty:
        print("\nWarning: df_countries is empty. Cannot calculate _LATEST_DF.")
    elif 'date' not in df_countries.columns: # Check if 'date' column is missing
        print("\nWarning: 'date' column missing in df_countries. Cannot calculate _LATEST_DF.")

# 4. Exploratory Data Analysis (EDA)
print("\n4. Exploratory Data Analysis (EDA)")
print("-" * 50)

try:
    # Total cases over time
    plt.figure(figsize=(14, 8))
    for country in df_countries['location'].unique():
        country_data = df_countries[df_countries['location'] == country]
        plt.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2)
    plt.title('Total COVID-19 Cases Over Time', fontsize=20)
    plt.xlabel('Date'); plt.ylabel('Total Cases')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # Total deaths over time
    plt.figure(figsize=(14, 8))
    for country in df_countries['location'].unique():
        country_data = df_countries[df_countries['location'] == country]
        plt.plot(country_data['date'], country_data['total_deaths'], label=country, linewidth=2)
    plt.title('Total COVID-19 Deaths Over Time', fontsize=20)
    plt.xlabel('Date'); plt.ylabel('Total Deaths')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # Death rate
    plt.figure(figsize=(14, 8))
    for country in df_countries['location'].unique():
        data = df_countries[df_countries['location'] == country].copy()
        data['death_rate'] = np.where(data['total_cases'] > 0, 
                                      data['total_deaths'] / data['total_cases'] * 100, 0)
        plt.plot(data['date'], data['death_rate'], label=country, linewidth=2)
    plt.title('COVID-19 Death Rate Over Time (%)', fontsize=20)
    plt.xlabel('Date'); plt.ylabel('Death Rate (%)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # Bar charts - latest data
    if not _LATEST_DF.empty:
        if 'total_cases' in _LATEST_DF.columns:
            plt.figure(figsize=(14, 8))
            sns.barplot(x='location', y='total_cases', data=_LATEST_DF.sort_values('total_cases', ascending=False))
            plt.title('Total Cases by Country (Latest Data)'); plt.xlabel('Country'); plt.ylabel('Total Cases')
            plt.xticks(rotation=45); plt.tight_layout(); plt.grid(axis='y', alpha=0.3)
            plt.show()
        else:
            print("Skipping 'Total Cases by Country' bar chart: 'total_cases' not in _LATEST_DF.")

        if 'total_deaths' in _LATEST_DF.columns:
            plt.figure(figsize=(14, 8))
            sns.barplot(x='location', y='total_deaths', data=_LATEST_DF.sort_values('total_deaths', ascending=False))
            plt.title('Total Deaths by Country (Latest Data)'); plt.xlabel('Country'); plt.ylabel('Total Deaths')
            plt.xticks(rotation=45); plt.tight_layout(); plt.grid(axis='y', alpha=0.3)
            plt.show()
        else:
            print("Skipping 'Total Deaths by Country' bar chart: 'total_deaths' not in _LATEST_DF.")
    else:
        print("Skipping latest data bar charts as _LATEST_DF is empty.")

    print("Basic EDA visualizations completed")
except Exception as e:
    print(f"Error during EDA visualization: {e}")

# 5. Vaccination Analysis
print("\n5. Vaccination Analysis")
print("-" * 50)

try:
    if 'people_vaccinated' in df_countries.columns: # Check df_countries for the line plot
        plt.figure(figsize=(14, 8))
        for country in df_countries['location'].unique():
            country_data = df_countries[df_countries['location'] == country]
            plt.plot(country_data['date'], country_data['people_vaccinated'], label=country, linewidth=2)
        plt.title('Vaccination Progress Over Time (People Vaccinated)')
        plt.xlabel('Date'); plt.ylabel('People Vaccinated')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()

        if not _LATEST_DF.empty and 'population' in _LATEST_DF.columns and 'people_vaccinated' in _LATEST_DF.columns:
            _LATEST_DF['vax_percentage'] = np.where(
                _LATEST_DF['population'].fillna(0) > 0,
                (_LATEST_DF['people_vaccinated'].fillna(0) / _LATEST_DF['population']) * 100,
                0
            )
            _LATEST_DF['vax_percentage'] = _LATEST_DF['vax_percentage'].fillna(0)

            if _LATEST_DF['vax_percentage'].notna().any():
                plt.figure(figsize=(14, 8))
                sns.barplot(x='location', y='vax_percentage', data=_LATEST_DF.sort_values('vax_percentage', ascending=False))
                plt.title('Vaccination Rate by Country (%) (Latest Data)'); plt.xlabel('Country'); plt.ylabel('Vaccinated (%)')
                plt.xticks(rotation=45); plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

            # Pie chart for a specific country (e.g., United States)
            country_for_pie = 'United States'
            us_latest_data = _LATEST_DF[_LATEST_DF['location'] == country_for_pie]
            if not us_latest_data.empty:
                population_val = us_latest_data['population'].iloc[0]
                people_vaccinated_val = us_latest_data['people_vaccinated'].iloc[0]

                if pd.notna(population_val) and pd.notna(people_vaccinated_val) and population_val > 0:
                    vaccinated_count = min(people_vaccinated_val, population_val)
                    vaccinated_count = max(0, vaccinated_count) # Ensure not negative
                    unvaccinated_count = population_val - vaccinated_count

                    if unvaccinated_count >= 0:
                        plt.figure(figsize=(8, 8))
                        plt.pie([vaccinated_count, unvaccinated_count],
                                labels=['Vaccinated', 'Unvaccinated'],
                                autopct='%1.1f%%',
                                colors=['#28a745', '#dc3545'],
                                startangle=90)
                        plt.title(f'Vaccination Status in {country_for_pie} (Latest Data)')
                        plt.axis('equal'); plt.tight_layout(); plt.show()
                    else:
                        print(f"Could not generate pie chart for {country_for_pie}: inconsistent data (unvaccinated < 0).")
                else:
                    print(f"Insufficient or invalid data for pie chart in {country_for_pie} (Population: {population_val}, People Vaccinated: {people_vaccinated_val}).")
            else:
                print(f"No latest data found for {country_for_pie} to generate pie chart.")
        else:
            print("Skipping vaccination rate bar chart and pie chart: _LATEST_DF is empty or 'population'/'people_vaccinated' columns missing.")
        print("Vaccination analysis visualizations completed")
    else:
        print("Vaccination data ('people_vaccinated') not available in df_countries for line plot.")
except Exception as e:
    print(f"Error during vaccination visualization: {e}")

# 6. Key Insights and Report
print("\n6. Key Insights and Report")
print("-" * 50)

try:
    if not _LATEST_DF.empty:
        # 1. Highest total cases
        if 'total_cases' in _LATEST_DF.columns and _LATEST_DF['total_cases'].notna().any():
            top_cases_country = _LATEST_DF.loc[_LATEST_DF['total_cases'].idxmax()]
            print(f"1. Highest total cases: {top_cases_country['location']} - {top_cases_country['total_cases']:,.0f}")
        else:
            print("1. Could not determine highest total cases: 'total_cases' missing or all NaN in _LATEST_DF.")

        # 2. Highest death rate
        if 'total_deaths' in _LATEST_DF.columns and 'total_cases' in _LATEST_DF.columns:
            _LATEST_DF['death_rate'] = np.where(
                _LATEST_DF['total_cases'].fillna(0) > 0,
                (_LATEST_DF['total_deaths'].fillna(0) / _LATEST_DF['total_cases']) * 100,
                0
            )
            _LATEST_DF['death_rate'] = _LATEST_DF['death_rate'].fillna(0)
            if _LATEST_DF['death_rate'].notna().any(): # idxmax needs at least one non-NaN value
                top_death_rate_country = _LATEST_DF.loc[_LATEST_DF['death_rate'].idxmax()]
                print(f"2. Highest death rate: {top_death_rate_country['location']} - {top_death_rate_country['death_rate']:.2f}%")
            else: # Should only happen if _LATEST_DF is empty or death_rate column is all NaNs before fillna(0)
                 print("2. Could not determine highest death rate (all values are NaN).")
        else:
            print("2. Could not calculate death rate: 'total_deaths' or 'total_cases' missing in _LATEST_DF.")

        # 3. Highest vaccination rate
        if 'people_vaccinated' in _LATEST_DF.columns and 'population' in _LATEST_DF.columns:
            _LATEST_DF['vax_rate'] = np.where(
                _LATEST_DF['population'].fillna(0) > 0,
                (_LATEST_DF['people_vaccinated'].fillna(0) / _LATEST_DF['population']) * 100,
                0
            )
            _LATEST_DF['vax_rate'] = _LATEST_DF['vax_rate'].fillna(0)
            if _LATEST_DF['vax_rate'].notna().any():
                top_vax_country = _LATEST_DF.loc[_LATEST_DF['vax_rate'].idxmax()]
                print(f"3. Highest vaccination rate: {top_vax_country['location']} - {top_vax_country['vax_rate']:.2f}%")
            else:
                print("3. Could not determine highest vaccination rate (all values are NaN).")
        else:
            print("3. Could not calculate vaccination rate: 'people_vaccinated' or 'population' missing in _LATEST_DF.")

        # 4. Highest average new cases (last 30 days)
        if 'new_cases' in df_countries.columns and 'date' in df_countries.columns:
            max_date = df_countries['date'].max()
            if pd.notna(max_date):
                last_month_df = df_countries[df_countries['date'] >= max_date - pd.Timedelta(days=30)]
                if not last_month_df.empty:
                    avg_new_cases = last_month_df.groupby('location')['new_cases'].mean().dropna()
                    if not avg_new_cases.empty:
                        top_avg_new_location = avg_new_cases.idxmax()
                        top_avg_new_value = avg_new_cases.max()
                        print(f"4. Highest average new cases (last 30 days): {top_avg_new_location} - {top_avg_new_value:.2f}")
                    else:
                        print("4. Could not determine highest average new cases (last 30 days): No valid mean data after dropping NaNs.")
                else:
                    print("4. Could not determine highest average new cases (last 30 days): No data in the last 30 days.")
            else:
                print("4. Could not determine highest average new cases (last 30 days): Max date not available.")
        else:
            print("4. Could not calculate average new cases: 'new_cases' or 'date' missing in df_countries.")

        # 5. Highest deaths per million
        if 'total_deaths_per_million' in _LATEST_DF.columns and _LATEST_DF['total_deaths_per_million'].notna().any():
            top_deaths_per_million_country = _LATEST_DF.loc[_LATEST_DF['total_deaths_per_million'].idxmax()]
            print(f"5. Highest deaths per million: {top_deaths_per_million_country['location']} - {top_deaths_per_million_country['total_deaths_per_million']:.2f}")
        else:
            print("5. Could not determine highest deaths per million: 'total_deaths_per_million' missing or all NaN in _LATEST_DF.")
    else:
        print("Key insights cannot be generated because _LATEST_DF is empty.")

    print("\nCOVID-19 Data Analysis Complete!")
except Exception as e:
    print(f"Error generating insights: {e}")

print("\nComprehensive Analysis Summary")
print("=" * 50)
print(f"""
This analysis of COVID-19 data across selected countries reveals several key patterns:

1. Case Distribution: The pandemic has affected countries differently.
2. Mortality Patterns: Death rates vary due to healthcare, demographics, and reporting.
3. Vaccination Impact: Higher vaccination = better outcomes.
4. Time Trends: Countries saw different waves and timing.
5. Healthcare Impact: Stronger systems reduce mortality.

This provides a strong data-driven foundation for public health planning.
The data spans from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}.
""")
