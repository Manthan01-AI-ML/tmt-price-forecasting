"""
03_modeling_prophet_arima_xgb.py
Compare Prophet, ARIMA, and XGBoost forecasts.
"""

import pandas as pd
import numpy as np
import math
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

def safe_eval(y_true, y_pred, label="model"):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{label} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"{label} evaluation failed: {e}")

df = pd.read_csv("../data/features_tmt.csv", parse_dates=["Invoice_Date"])
df = df.sort_values("Invoice_Date")

split_date = df["Invoice_Date"].max() - pd.DateOffset(months=3)
train = df[df["Invoice_Date"] <= split_date]
test = df[df["Invoice_Date"] > split_date]

# Prophet + Quantity_MT
try:
    tmp = train[["Invoice_Date", "Unit_Price", "Quantity_MT"]].copy()
    tmp["ds"] = tmp["Invoice_Date"].dt.floor("D")
    tmp["y"] = tmp["Unit_Price"]

    m_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m_prophet.add_regressor("Quantity_MT")
    m_prophet.fit(tmp[["ds", "y", "Quantity_MT"]])

    test_ = test.copy()
    test_["Invoice_Date"] = test_["Invoice_Date"].dt.floor("D")
    test_["Quantity_MT"] = test_["Quantity_MT"].fillna(tmp["Quantity_MT"].median())

    future = test_[["Invoice_Date", "Quantity_MT"]].rename(columns={"Invoice_Date": "ds"})
    fc = m_prophet.predict(future)[["ds", "yhat"]]

    dfp = fc.set_index("ds").join(test.set_index("Invoice_Date"))[["Unit_Price", "yhat"]].dropna()
    safe_eval(dfp["Unit_Price"], dfp["yhat"], "Prophet+Qty")
except Exception as e:
    print("Prophet failed:", e)

# ARIMA
try:
    y_train = train.set_index("Invoice_Date")["Unit_Price"]
    y_test = test.set_index("Invoice_Date")["Unit_Price"]

    model_arima = ARIMA(y_train, order=(5,1,0))
    model_fit = model_arima.fit()
    fc_arima = model_fit.forecast(steps=len(y_test))

    safe_eval(y_test, fc_arima, "ARIMA")
except Exception as e:
    print("ARIMA failed:", e)

# --- XGBoost (numeric-only features to avoid categorical errors) ---
try:
    # keep only numeric/bool columns; drop target/date
    numeric_cols = train.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "Unit_Price"]

    # quick sanity
    if len(feature_cols) == 0:
        raise ValueError("No numeric features found for XGBoost.")

    X_train, y_train_xgb = train[feature_cols].fillna(0), train["Unit_Price"]
    X_test,  y_test_xgb  = test[feature_cols].fillna(0),  test["Unit_Price"]

    model_xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model_xgb.fit(X_train, y_train_xgb)
    preds_xgb = model_xgb.predict(X_test)

    safe_eval(y_test_xgb, preds_xgb, "XGBoost")
except Exception as e:
    print("XGBoost failed:", e)
    # Helpful debug: show offending dtypes once
    non_numeric = [f"{c}:{train[c].dtype}" for c in train.columns
                   if train[c].dtype not in ("int64","float64","bool","int32","float32")]
    print("Non-numeric columns present:", non_numeric)
