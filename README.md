
```markdown
# 🌍 COVID-19 Global Data Tracker

This project analyzes global trends in COVID-19 cases, deaths, and vaccinations using publicly available data. It includes visualizations and key insights to understand the progression of the pandemic across selected countries.

## 📁 Dataset

The analysis is based on the [Our World in Data COVID-19 dataset](https://ourworldindata.org/coronavirus-source-data), saved locally as:

```

owid-covid-data.csv

````

Make sure this file is in the same directory as the notebook/script.

## 📊 Features

- Loads and explores global COVID-19 data.
- Cleans and prepares data for analysis.
- Visualizes:
  - Total cases and deaths over time.
  - Death rate trends (deaths/cases).
  - Latest totals via bar charts.
  - Vaccination progress and vaccination percentage.
- Identifies key insights from the latest data.

## 🛠️ Requirements

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn
````

### Python Version

Tested on **Python 3.8+**

## 🚀 How to Run

1. Ensure you have the required dataset (`owid-covid-data.csv`) in the working directory.
2. Open the notebook or run the Python script:

   ```bash
   jupyter notebook covid_data_tracker.ipynb
   ```

   or

   ```bash
   python covid_data_tracker.py
   ```

## 🌐 Countries Included in Analysis

The following countries are included in the focused analysis:

* United States
* India
* Brazil
* Kenya
* United Kingdom
* Germany

## 📈 Example Visualizations

* **Total COVID-19 Cases Over Time**
* **Total Deaths Over Time**
* **Death Rate (%) Trends**
* **Vaccination Coverage (%)**

## 🧠 Key Insights (Sample)

* Country with highest total cases
* Country with highest death rate
* Countries with highest vaccination coverage

> Note: Insights are generated using the most recent available data per country.

## 📌 License

This project is for educational and research purposes only. Data is sourced from [Our World in Data](https://ourworldindata.org/coronavirus).

## 🙏 Acknowledgements

* Our World in Data for the comprehensive COVID-19 dataset.
* The global healthcare community for their continued efforts in battling the pandemic.
