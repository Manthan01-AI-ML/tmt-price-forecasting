# 📊 TMT Steel Price Forecasting — Procurement Analytics

**Author:** Manthan Teotia
**Role:** Associate ML Engineer

---

## 📌 Project Overview
This project focuses on **forecasting purchase prices** of TMT steel to help procurement teams:
- Negotiate better with suppliers
- Plan budgets more accurately
- Understand SKU-level (diameter & grade) price trends

The solution uses **time series forecasting models**:
1. **Prophet** — for capturing seasonality and trend
2. **ARIMA** — for statistical baseline forecasting
3. **XGBoost** — for feature-rich machine learning forecasting

---

## 📂 Repository Structure
── data/
│ ├── README.md # Dataset details
│ ├── 3year_tmt_data.csv 
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Data loading & preprocessing
│ ├── 02_feature_engineering.ipynb# Time features, lags, rolling means, vendor encoding
│ ├── 03_modeling_prophet_arima_xgb.ipynb # Modeling & evaluation
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── modeling.py
│
├── results/
│ ├── forecast_TMTdata.xlsx # Example forecast output
│
├── requirements.txt
└── README.md


---

## 🛠 Tech Stack
- **Python**: 3.11
- **Libraries**:
  - pandas, numpy, matplotlib, scikit-learn
  - prophet, statsmodels, xgboost
  - openpyxl (Excel file handling)

---

## 🔍 Workflow
1. **Data Cleaning** — Removing missing values, ensuring date formats, fixing typos.
2. **Feature Engineering**:
   - Calendar features (month, quarter, day-of-week)
   - Lag features (1, 7, 30 days)
   - Rolling averages
   - Vendor & order quantity encoding
3. **Model Training**:
   - Prophet — with quantity as regressor
   - ARIMA — as statistical benchmark
   - XGBoost — with time & lag features
4. **Evaluation**:
   - Metrics: MAE, RMSE
   - Prophet + Quantity: MAE ~ ₹2,520 | RMSE ~ ₹3,556
   - XGBoost: MAE ~ ₹2,046 | RMSE ~ ₹2,966
5. **Reporting**:
   - SKU-level day-wise forecast for 30 days
   - Monthly averages for procurement planning

## 📊 Business Value
- SKU & vendor-specific price forecasting
- Insights into seasonal & vendor-driven price variation
- Helps procurement teams lock in better rates

---

## ⚠ Disclaimer
The dataset used here is **synthetic** and generated for educational purposes only.  
Any resemblance to actual suppliers or transactions is coincidental.
