"""
Leakage-safe feature engineering and preprocessing for customer churn.

Run:
    python feature_engineering.py

Outputs:
    data/X_train.pkl
    data/X_test.pkl
    data/y_train.pkl
    data/y_test.pkl
    data/X_train_raw.csv
    data/X_test_raw.csv
    data/y_train_raw.csv
    data/y_test_raw.csv
    models/preprocessor.joblib
    models/feature_names.joblib

Important:
    All data-dependent transformations are fitted on the training split only.
    This keeps validation/test information out of the training process.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.20
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_COLUMNS = [
    "phone_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


# -----------------------------------------------------------------------------
# Row-level feature engineering only.
# No dataset-wide statistics are calculated here.
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic row-level churn features without target leakage."""
    out = df.copy()

    missing = [c for c in SERVICE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required service columns: {missing}")

    def is_active(series: pd.Series) -> pd.Series:
        return series.eq("Yes") | series.eq(1)

    out["num_services"] = sum(
        is_active(out[c]).astype("int8") for c in SERVICE_COLUMNS
    )

    tenure = pd.to_numeric(out["tenure_months"], errors="coerce")
    monthly = pd.to_numeric(out["monthly_charges"], errors="coerce")
    total = pd.to_numeric(out["total_charges"], errors="coerce")

    out["avg_monthly_spend"] = (
        total / tenure.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    out["charge_per_service"] = (
        monthly / out["num_services"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Keep this as a row-level feature. The old implementation used the
    # complete dataset's 75th percentile before splitting, which leaked test
    # distribution information into training.
    out["is_high_value"] = monthly

    out["tenure_group"] = pd.cut(
        tenure,
        bins=[-np.inf, 12, 24, 48, np.inf],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"],
    ).astype("object")

    tech = pd.to_numeric(out["num_tech_tickets"], errors="coerce")
    admin = pd.to_numeric(out["num_admin_tickets"], errors="coerce")
    out["ticket_total"] = tech + admin
    out["has_any_ticket"] = (out["ticket_total"] > 0).astype("int8")

    out["charge_x_tenure"] = monthly * tenure
    satisfaction = pd.to_numeric(out["satisfaction_score"], errors="coerce")
    out["low_satisfaction"] = (satisfaction < 3).astype("int8")

    return out


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer using training columns only."""
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names from a fitted ColumnTransformer."""
    return preprocessor.get_feature_names_out().tolist()


def run_feature_engineering(
    input_path: str = "data/customer_churn.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """Split first, then fit all preprocessing on training data only."""
    print("\n🔧 Running leakage-safe feature engineering …\n")

    df = pd.read_csv(input_path)
    required = {"customer_id", "churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = engineer_features(df)
    X = df.drop(columns=["customer_id", "churn"])
    y = pd.to_numeric(df["churn"], errors="raise").astype(int).to_numpy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = _feature_names(preprocessor)

    print(f"  Train size: {len(X_train_raw):,} | Test size: {len(X_test_raw):,}")
    print(f"  Train churn rate: {y_train.mean():.2%}")
    print(f"  Test churn rate : {y_test.mean():.2%}")
    print(f"  Encoded features: {len(feature_names):,}")

    joblib.dump(X_train, DATA_DIR / "X_train.pkl")
    joblib.dump(X_test, DATA_DIR / "X_test.pkl")
    joblib.dump(y_train, DATA_DIR / "y_train.pkl")
    joblib.dump(y_test, DATA_DIR / "y_test.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")

    X_train_raw.reset_index(drop=True).to_csv(DATA_DIR / "X_train_raw.csv", index=False)
    X_test_raw.reset_index(drop=True).to_csv(DATA_DIR / "X_test_raw.csv", index=False)
    pd.Series(y_train, name="churn").to_csv(DATA_DIR / "y_train_raw.csv", index=False)
    pd.Series(y_test, name="churn").to_csv(DATA_DIR / "y_test_raw.csv", index=False)

    print("\n✅ Preprocessor and train/test artifacts saved.")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    run_feature_engineering()