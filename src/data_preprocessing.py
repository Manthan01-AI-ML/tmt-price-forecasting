"""
01_data_cleaning.py
Basic cleanup of raw TMT steel purchase data.
"""

import pandas as pd
import numpy as np
import os

# === Load raw file ===
# Change path if your raw file is elsewhere
df_raw = pd.read_excel("../data/raw_tmt_data.xlsx", parse_dates=["Invoice_Date"])

# === Keep essential columns ===
cols_to_keep = [
    "Invoice_Date",
    "Material_Name",
    "Quantity_MT",
    "Total_Invoice_Value",
    "Unit_Price",
    "UOM",
    "Vendor_Name"
]
df = df_raw[cols_to_keep].copy()

# === Handle negatives (credit/debit notes) ===
df = df[df["Quantity_MT"] >= 0]

# === Ensure numeric ===
df["Quantity_MT"] = pd.to_numeric(df["Quantity_MT"], errors="coerce")
df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

# === Drop rows with missing price or date ===
df = df.dropna(subset=["Invoice_Date", "Unit_Price"])

# === Sort chronologically ===
df = df.sort_values("Invoice_Date")

# === Save cleaned ===
os.makedirs("../data", exist_ok=True)
df.to_csv("../data/cleaned_tmt_data.csv", index=False)

print("✅ Cleaning complete. Saved to ../data/cleaned_tmt_data.csv")
print(df.head())
