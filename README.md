# Project Configuration & Setup Guide

Welcome to the Air Quality and Weather Clustering Project! This guide provides a comprehensive overview of the project structure, dependencies, data flow, and instructions for running the clustering models. 

## 1. Libraries and Dependencies

Before running any scripts or notebooks, ensure you have the following Python libraries installed. You can install them via `pip`:

```bash
pip install pandas numpy scikit-learn altair requests openpyxl jupyter
```

**Core Libraries Used:**
- `pandas` & `numpy`: Data manipulation, aggregation, and mathematical operations.
- `requests`: Fetching raw data from external APIs.
- `datetime` & `time`: Managing time series data and API request intervals.
- `scikit-learn` (`StandardScaler`, `KMeans`): Used for data normalization and K-Means clustering.
- `altair`: Creating interactive data visualizations (saved as HTML).
- `openpyxl`: Reading and writing Excel (`.xlsx`) datasets.

---

## 2. Data Collection (Open-Meteo)

Data is dynamically fetched from the **Open-Meteo API** (both AQI and Weather archives). This process is handled in the `historical data fetch.ipynb` notebook.

- **Cities:** Karachi & Lahore
- **Timeframe:** From `2022-01-01` up to the present day.
- **Process:** The script makes concurrent calls to the AQI API and the Weather API. It extracts features like PM10, PM2.5, Carbon Monoxide, Temperature, Humidity, and Solar Radiation.
- **Output:** The merged raw data is saved into a multi-sheet Excel file: `pakistan_aqi_weather.xlsx` (with separate sheets for 'Karachi' and 'Lahore').

---

## 3. Data Cleaning and Transformation

Prior to modeling, the raw data undergoes several transformations to ensure model accuracy. In the clustering pipeline (`objective clusturing.py`), the following steps are executed:
- **Filtering:** Non-numeric and timestamp columns are separated.
- **Handling Missing Values:** Rows containing `NaN` (null) values in the required feature sets are dropped dynamically.
- **Normalization:** The features are scaled using Scikit-Learn's `StandardScaler` (Z-score normalization). This ensures that variables with different units (e.g., Temperature in °C vs. Pressure in hPa) contribute equally to the distance calculations in the K-Means algorithm.

*(Note: Additional exploratory data cleaning operations are maintained inside `data cleaning.ipynb`).*

---

## 4. How to Generate Clusters (Run File)

To generate the machine learning clusters and visualizations, you need to execute the main Python script.

**Execution Command:**
```bash
python "objective clusturing.py"
```

**What this script does:**
1. Loads the raw dataset `pakistan_aqi_weather.xlsx`.
2. Iterates through each city's sheet (Karachi and Lahore).
3. Applies K-Means clustering (K=5) across 7 distinct weather/AQI objectives.
4. Generates interactive HTML dashboards with Elbow charts, scatter plots, and feature heatmaps.
5. Saves the newly clustered data into separate Excel files.

---

## 5. Project File Structure

Here is a breakdown of the file types and their roles within this project:

### 🐍 Python Scripts
- `objective clusturing.py`: The main execution script for running K-Means clustering on the raw data.
- `cluster.py`: Additional modular clustering logic (if utilized).

### 📓 Jupyter Notebooks
- `historical data fetch.ipynb`: Script to pull raw data from the Open-Meteo API.
- `data cleaning.ipynb`: Sandbox for exploratory data analysis and cleaning.
- `kar model.ipynb` & `lah model.ipynb`: City-specific modeling experiments.
- `kar_aggregation.ipynb` & `lah_aggregation.ipynb`: Notebooks for temporal aggregations and grouping.

### 📊 Data Files (Excel)
- `pakistan_aqi_weather.xlsx`: The raw dataset generated from the API.
- `weather_clustered_Karachi.xlsx`: Karachi's dataset appended with objective cluster labels.
- `weather_clustered_Lahore.xlsx`: Lahore's dataset appended with objective cluster labels.
- `weather_clustered_by_objective.xlsx`: A consolidated view of clustered outputs.

### 🌐 Visualizations
- `viz_Karachi.html` & `viz_Lahore.html`: Interactive Altair dashboard visualizations depicting the clustering performance, data distribution, and feature importance.

---

## 6. Cluster Datasets & Objectives

The machine learning model generates **5 distinct clusters (K=5)** for every objective. The clusters represent the severity/intensity of the features:
- **0** = Very Low
- **1** = Low
- **2** = Moderate
- **3** = High
- **4** = Extreme

The model groups the data based on **7 Specific Objectives**, resulting in 7 new cluster columns per dataset. The objectives are:
1. **Rainfall** (precipitation, humidity, cloud cover, etc.)
2. **Cloud** (cloud covers at varying altitudes, aerosol depth)
3. **Solar** (radiation metrics, UV index)
4. **HeatWave** (temperature, apparent temp, wind speed)
5. **IndPollution** (PM10, PM2.5, NO2, CO, SO2, Dust)
6. **AQI** (Core air quality indicators)
7. **AirPollution** (Combination of AQI and wind metrics)

Each objective has its own dynamically generated visual dashboard in the resulting HTML files.
