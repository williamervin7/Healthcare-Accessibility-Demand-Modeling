![Python application](https://github.com/williamervin7/Healthcare-Accessibility-Demand-Modeling/actions/workflows/python-app.yml/badge.svg)

# Healthcare Accessibility Classification: Identifying "Healthcare Deserts" in Houston

## 📌 Project Overview
This project identifies underserved geographic areas (Healthcare Deserts) within the Houston, TX metropolitan area. By modeling the relationship between demographic demand indicators and healthcare facility density, this tool provides actionable insights for Healthcare Real Estate Investment Trusts (REITs) to optimize facility placement and investment.

## 🚦 Project Status: **EDA Phase Complete**
- [x] **Data Acquisition:** Integrated CMS Provider of Services and US Census ACS data.
- [x] **Data Engineering:** Resolved ZIP-code type mismatches and automated multi-feature imputation.
- [x] **Exploratory Analysis:** Identified non-linear relationships between affluence and facility access.
- [ ] **Modeling:** Custom Logistic Regression implementation (Next Step).

## 🎯 Objective
To classify Houston-area ZIP codes as "Underserved" or "Adequately Served" using a **Logistic Regression model implemented from scratch**.

## 🛠️ Technical Highlights (OMSA Readiness)
While standard libraries like `scikit-learn` are available, the core of this project features a **custom implementation of Logistic Regression** using `NumPy` to demonstrate:
* **Linear Algebra:** Matrix multiplication for feature sets ($X$) and weight vectors ($\theta$).
* **Calculus:** Partial derivatives and Gradient Descent for cost minimization.
* **Statistical Theory:** Sigmoid activation function and Binary Cross-Entropy (Log-Loss).

## 🔍 Key EDA Findings
Initial exploration of the Texas metropolitan area revealed:
* **The Income-Density Paradox:** Contrary to the initial hypothesis, higher-income suburban ZIP codes in Texas often exhibit *lower* facility density, while lower-income urban cores show higher density due to centralized medical infrastructure.
* **Aging Vulnerability:** A negative correlation exists between median age and facility density, suggesting that older populations may face the highest barriers to local care.

## 📊 Data Sources
The analysis joins two primary datasets at the ZIP code (ZCTA) level:
1.  **Supply Data:** CMS Provider of Services (POS) File — comprehensive infrastructure data on hospitals, clinics, and rural health centers.
2.  **Demand Data:** U.S. Census Bureau ACS 5-Year Estimates — Median Age, Total Population, Household Income, and secondary factors (Uninsured/Vehicle Access).

## 🏗️ Project Structure
```text
├── data/                  
├── notebooks/   
|   ├── eda.ipynb         # Started_eda.ipynb (Data Cleaning & Hypo Testing)
├── src/
│   ├── data_acquisition.py # API calls and CMS data ingestion
│   ├── preprocessing.py    # Merging and SettingWithCopy-safe cleaning
│   ├── modeling.py         # NumPy-based Logistic Regression
│   └── visualizations.py   # Regression plots and Seaborn heatmaps
├── test/
|   ├── test_model.py/
├── README.md
└── requirements.txt
```

## 🚀 Future Results & Visualizations
* **Classification Metrics:** Precision-Recall curves and F1-Score comparisons between the custom model and `scikit-learn`.
* **Geospatial Analysis:** Folium/Geopandas choropleth maps identifying priority investment ZIP codes.

## 🧑‍💻 Author
**William Ervin** *Data Analytics Capstone Project* *Target: Georgia Tech OMSA Admissions Portfolio*
***

