"""
02_feature_engineering.py
Build features for modeling.
"""

import sys
import os
sys.path.append(os.path.abspath(".."))

import pandas as pd
from src import feature_engineering as fe

df = pd.read_csv("../data/cleaned_tmt_data.csv", parse_dates=["Invoice_Date"])
df = df.sort_values("Invoice_Date")

# Build features in one go
df_feat = fe.build_feature_matrix(df)

# Keep Quantity_MT even if NaN
df_feat = df_feat[df_feat["Unit_Price"].notna()]

os.makedirs("../data", exist_ok=True)
df_feat.to_csv("../data/features_tmt.csv", index=False)

print("✅ Feature engineering complete. Saved to ../data/features_tmt.csv")
print(df_feat.head()[["Invoice_Date", "Unit_Price", "Quantity_MT", "vendor_id"]])
