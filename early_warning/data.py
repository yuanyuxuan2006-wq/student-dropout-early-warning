"""Loading the UCI dataset, building the leakage-free feature table and splitting it.

Dataset: UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2021),
CC BY 4.0. data/data.csv is read from disk and downloaded from UCI if it is missing.
"""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # stratified splitting

from .config import DATA_PATH, DATA_URL, SEED


def load_raw_data(path=DATA_PATH):
    """Load the raw UCI table, downloading it first if it is not on disk."""
    if not path.exists():
        print("data.csv not found - downloading from UCI ...")
        with urllib.request.urlopen(DATA_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read("data.csv"))
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # One header ("Daytime/evening attendance\t") has a stray tab, so strip all names.
    df.columns = df.columns.str.strip()
    return df


def build_features(raw):
    """Return (features_df, y_all) using only information available at the end of semester 1."""
    # 1) Binary label: Dropout (1) vs Not Dropout (0 = Graduate + Enrolled)
    y_all = (raw["Target"] == "Dropout").astype(int).to_numpy()

    # 2) Remove the six semester-2 columns: they are recorded AFTER the advisor's decision
    #    point (end of semester 1), so using them would leak future information.
    sem2_cols = [c for c in raw.columns if "2nd sem" in c]
    assert len(sem2_cols) == 6, sem2_cols
    features_df = raw.drop(columns=sem2_cols + ["Target"])
    assert not any("2nd sem" in c for c in features_df.columns), "leakage check failed"

    # 3) Engineered semester-1 features (also used by the rule baseline).
    #    Edge case: students enrolled in 0 units have no pass rate; give them 0 and a flag.
    enrolled_1 = features_df["Curricular units 1st sem (enrolled)"]
    approved_1 = features_df["Curricular units 1st sem (approved)"]
    features_df["sem1_no_units"] = (enrolled_1 == 0).astype(int)
    features_df["sem1_pass_rate"] = (approved_1 / enrolled_1.replace(0, np.nan)).fillna(0.0)
    return features_df, y_all


def make_splits(X_df, y, seed=SEED):
    """Stratified 60/20/20 train/validation/test split (keeps the ~32% dropout rate)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)  # 0.25 * 0.8 = 0.2
    return X_train, X_val, X_test, y_train, y_val, y_test
