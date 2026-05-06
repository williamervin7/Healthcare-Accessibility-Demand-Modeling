# Healthcare Accessibility Classification: Identifying "Healthcare Deserts" in Houston

## 📌 Project Overview
This project identifies underserved geographic areas (Healthcare Deserts) within the Houston, TX metropolitan area. By modeling the relationship between demographic demand indicators and healthcare facility density, this tool provides actionable insights for Healthcare Real Estate Investment Trusts (REITs) to optimize facility placement and investment.

## 🎯 Objective
To classify Houston-area ZIP codes as "Underserved" or "Adequately Served" using a **Logistic Regression model implemented from scratch**.

## 🛠️ Technical Highlights (OMSA Readiness)
While standard libraries like `scikit-learn` are available, the core of this project features a **custom implementation of Logistic Regression** using `NumPy` to demonstrate a deep understanding of:
* **Linear Algebra:** Matrix multiplication for feature sets and weights.
* **Calculus:** Gradient Descent optimization for cost minimization.
* **Statistical Theory:** Log-Loss (Binary Cross-Entropy) and the Sigmoid activation function.

## 📊 Data Sources
The analysis joins two primary datasets at the ZIP code (ZCTA) level:
1.  **Supply Data:** CMS Provider of Services (POS) File — comprehensive data on hospitals, clinics, and healthcare infrastructure.
2.  **Demand Data:** U.S. Census Bureau ACS 5-Year Estimates — Median Age, Total Population, and Median Household Income.

## 🏗️ Project Structure
```text
├── data/                   # Raw and processed CSV files
├── src/
│   ├── data_acquisition.py # API calls (Census) and CMS data ingestion
│   ├── preprocessing.py    # Data cleaning, merging, and feature scaling
│   ├── modeling.py         # Custom Logistic Regression vs Sklearn implementation
│   ├── visualizations.py   # Geographic maps and diagnostic plots
│   └── main.py             # Orchestrates the full pipeline
├── notebooks/              # Exploratory Data Analysis (EDA)
├── README.md
└── requirements.txt
```

## 🚀 Key Results & Visualizations
* **Classification Metrics:** Accuracy, F1-Score, and ROC-AUC curve comparisons between the custom model and `scikit-learn`.
* **Geospatial Analysis:** Choropleth maps identifying priority ZIP codes for healthcare expansion in Houston.
* **Feature Importance:** Insights into how income and age significantly drive the probability of an area being a "healthcare desert."

## 🧑‍💻 Author
**William Ervin** *Data Analytics Capstone Project* *Aimed at Georgia Tech OMSA Admissions Portfolio*

