# Corporate Energy Spike Prediction & Carbon Mitigation Pipeline

An end-to-end data science pipeline built to predict, explain, and financially quantify anomalous "spikes" in corporate natural gas consumption. By merging internal energy records with external historical weather API data, this model serves as a predictive warning system to mitigate energy waste, lower greenhouse gas emissions, and optimize carbon tax savings.

---

## 🚀 Key Features

* **Automated Data Engineering:** Fetches historical weather data using coordinates via the Open-Meteo API and derives monthly Heating Degree Days (HDD) with an 18°C baseline.
* **Hybrid Modeling Pipeline:** Generates baseline time-series forecasts using **Meta Prophet** and injects the predictions (`yhat`) as a structural feature into a downstream **XGBoost Classifier**.
* **Time-Series Cross-Validation:** Implements a strict rolling `TimeSeriesSplit` (5 folds) to ensure chronological integrity and prevent data leakage.
* **Financial Impact Quantization:** Features an integrated Carbon Credit Calculator that maps predicted consumption spikes to actual financial tax liabilities using an explicit emissions factor ($0.05 \text{ Tons } CO_2e / \text{ GJ}$) and carbon market pricing.
* **Explainable AI (XAI):** Implements **SHAP (SHapley Additive exPlanations)** both globally (Summary plots) and locally (Waterfall plots) to expose the root causes driving specific energy spikes.
* **Automated Actionable Alerts:** Generates contextual alerts outlining real-time mitigation tasks and their corresponding tax savings value.

---

## 🛠️ Tech Stack & Dependencies

* **Data Wrangling:** `pandas`, `numpy`
* **API Integration:** `requests`
* **Time-Series Forecasting:** `prophet`
* **Machine Learning Framework:** `scikit-learn`, `xgboost`
* **Model Explainability:** `shap`
* **Visualization:** `matplotlib`

---

## 📊 Data & Architecture Pipeline

### 1. Data Ingestion & Target Definition
* **Energy Consumption:** Ingests monthly facility metrics, isolates natural gas workflows, and dynamically calculates a "Spike Threshold" defined as consumption exceeding the **90th percentile** for that specific facility.
* **Weather Integration:** Automatically calls the Open-Meteo API for target coordinates (Calgary) to extract daily mean temperatures, converting them into localized Heating Degree Days (HDD).

### 2. Feature Engineering & Preprocessing
* A `ColumnTransformer` handles categorical variables via structural `OneHotEncoder` pipelines while tracking feature name origins for full transparency.
* Prophet-driven time-series baselines capture long-term macro trends and annual seasonality.

### 3. Model Training & Evaluation
The model optimizes for precision-recall balance using an `aucpr` loss curve, yielding strong robustness on highly imbalanced anomalies:
* **Average ROC-AUC:** `0.8934`
* **Average PR-AUC:** `0.4038`

---
