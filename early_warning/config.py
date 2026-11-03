"""Shared constants: random seed, data location and column groups."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_URL = ("https://archive.ics.uci.edu/static/public/697/"
            "predict+students+dropout+and+academic+success.zip")

# Columns that are numeric codes for categories (not quantities) -> one-hot encoded.
CATEGORICAL_COLS = ["Marital status", "Application mode", "Course", "Previous qualification",
                    "Nacionality", "Mother's qualification", "Father's qualification",
                    "Mother's occupation", "Father's occupation"]

# Sensitive features (H1): gender, nationality, age, debtor, tuition fees, scholarship.
SENSITIVE_COLS = ["Gender", "Nacionality", "International", "Age at enrollment",
                  "Debtor", "Tuition fees up to date", "Scholarship holder"]
