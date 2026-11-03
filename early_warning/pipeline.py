"""End-to-end pipeline, verification against scikit-learn and per-student explanations."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression   # used ONLY to verify my from-scratch model

from .config import CATEGORICAL_COLS, SEED
from .data import build_features, load_raw_data, make_splits
from .metrics import tune_threshold
from .model import LogisticRegressionGD
from .preprocessing import Preprocessor


@dataclass
class Dataset:
    """Everything the experiments share: raw tables, splits and preprocessed matrices."""
    raw: pd.DataFrame
    features_df: pd.DataFrame
    y_all: np.ndarray
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    prep: Preprocessor
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    @property
    def feature_names(self):
        return self.prep.feature_names_


def load_dataset(seed=SEED, verbose=True):
    """Load, clean, split and preprocess the data."""
    raw = load_raw_data()
    features_df, y_all = build_features(raw)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = make_splits(features_df, y_all, seed)
    prep = Preprocessor(CATEGORICAL_COLS).fit(X_train_df)
    X_train, X_val, X_test = (prep.transform(d) for d in (X_train_df, X_val_df, X_test_df))
    data = Dataset(raw, features_df, y_all, X_train_df, X_val_df, X_test_df,
                   y_train, y_val, y_test, prep, X_train, X_val, X_test)

    if verbose:
        print("Raw shape:", raw.shape, "| missing values:", int(raw.isna().sum().sum()))
        print(raw["Target"].value_counts().to_string(), "\n")
        print(f"Features after removing semester 2: {features_df.shape[1]} raw columns "
              f"({len(CATEGORICAL_COLS)} categorical)")
        print(f"Dropout rate: {y_all.mean():.3f}")
        for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
            print(f"{name:5s}: {len(ys):4d} students, dropout rate {ys.mean():.3f}")
        print("Model input dimension after one-hot encoding:", X_train.shape[1])
    return data


def sklearn_equivalent(l2, n_train):
    """sklearn minimises C*sum(log loss) + 0.5||w||^2, which equals my objective when
    C = 1 / (l2 * n). Very weak regularisation is used if l2 = 0."""
    C = 1.0 / (l2 * n_train) if l2 > 0 else 1e6
    return LogisticRegression(C=C, solver="lbfgs", max_iter=10_000, tol=1e-10)


def top_contributors(model, x_row, names, k=3):
    """Features that pushed THIS student's risk up the most (weight * standardised value).

    Each name gets "(high)" or "(low)" to say whether the student is above or below the
    training average, e.g. "sem1_pass_rate (low)". One-hot columns ("Course=171") are only
    listed when the student IS in that category, so they need no suffix.
    """
    contributions = model.w * x_row
    order = np.argsort(-contributions)[:k]
    reasons = []
    for i in order:
        if contributions[i] <= 0:
            continue
        label = names[i] if "=" in names[i] else f"{names[i]} ({'high' if x_row[i] > 0 else 'low'})"
        reasons.append((label, round(float(contributions[i]), 3)))
    return reasons


def fit_pipeline(X_tr_df, X_va_df, y_tr, y_va, lr, l2, exclude_cols=()):
    """Preprocess -> train from scratch -> tune threshold on validation. Returns everything
    needed to score new students with the same preprocessing."""
    pp = Preprocessor(CATEGORICAL_COLS, exclude_cols=exclude_cols).fit(X_tr_df)
    Xtr, Xva = pp.transform(X_tr_df), pp.transform(X_va_df)
    mdl = LogisticRegressionGD(lr=lr, l2=l2).fit(Xtr, y_tr, Xva, y_va)
    thr, _ = tune_threshold(y_va, mdl.predict_proba(Xva))
    return pp, mdl, thr
