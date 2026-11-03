"""One-hot encoding and standardisation learned from the training split only."""
import numpy as np
import pandas as pd


class Preprocessor:
    """One-hot encodes categorical codes and standardises every column.

    All statistics (kept category levels, means, standard deviations) are learned from the
    TRAINING split only, so no information from validation/test leaks into training.
    Rare category levels (< min_count training students) are merged into "other" so the
    model does not learn weights from a handful of students.
    """

    def __init__(self, categorical_cols, exclude_cols=(), min_count=30):
        self.exclude_cols = list(exclude_cols)
        self.categorical_cols = [c for c in categorical_cols if c not in self.exclude_cols]
        self.min_count = min_count

    def _encode(self, df):
        df = df.drop(columns=self.exclude_cols)
        parts = [df.drop(columns=self.categorical_cols).astype(float)]
        for col in self.categorical_cols:
            values = df[col].where(df[col].isin(self.levels_[col]), other=-1)  # -1 = "other"
            for level in self.levels_[col] + [-1]:
                name = f"{col}={'other' if level == -1 else level}"
                parts.append((values == level).astype(float).rename(name))
        return pd.concat(parts, axis=1)

    def fit(self, train_df):
        self.levels_ = {}
        for col in self.categorical_cols:
            counts = train_df[col].value_counts()
            self.levels_[col] = sorted(counts[counts >= self.min_count].index.tolist())
        encoded = self._encode(train_df)
        self.feature_names_ = encoded.columns.tolist()
        self.mean_ = encoded.mean().to_numpy()
        std = encoded.std(ddof=0).to_numpy()
        self.std_ = np.where(std > 0, std, 1.0)  # constant column -> avoid divide-by-zero
        return self

    def transform(self, df):
        encoded = self._encode(df).reindex(columns=self.feature_names_, fill_value=0.0)
        return (encoded.to_numpy() - self.mean_) / self.std_
