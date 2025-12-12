"""
Regenerate scalers.pkl using the local sklearn version.
Run this once after updating sklearn or when moving between environments.
"""
import glob
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Configuration - MUST match training notebook
FEATURE_COLUMNS = [
    "Local_X", "Local_Y", "v_Vel", "v_Acc", "Space_Headway", "dis_cen",
    "i_l", "i_r", "i_f", "dis_l", "dis_r", "dis_f"
]

# Paths
RAW_DATA_DIR = "data/raw"
OUTPUT_PATH = "data/processed/scalers.pkl"


def load_all_csv(directory: str) -> pd.DataFrame:
    """Load and concatenate all CSVs in the directory."""
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=FEATURE_COLUMNS)
        except ValueError:
            df = pd.read_csv(f)
            df = df[FEATURE_COLUMNS]
        frames.append(df)
    
    return pd.concat(frames, ignore_index=True)


def main():
    print(f"Loading data from {RAW_DATA_DIR}...")
    df = load_all_csv(RAW_DATA_DIR)
    print(f"Total rows: {len(df)}")
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Fit scaler
    print("Fitting MinMaxScaler...")
    scaler = MinMaxScaler()
    scaler.fit(df.values.astype(np.float32))
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(scaler, OUTPUT_PATH)
    print(f"✅ Scaler saved to {OUTPUT_PATH}")
    print(f"   data_min_: {scaler.data_min_[:2]} (first 2)")
    print(f"   data_max_: {scaler.data_max_[:2]} (first 2)")


if __name__ == "__main__":
    main()
