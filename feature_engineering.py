"""
feature_engineering.py
=======================
Handles:
  - Missing value imputation
  - Categorical encoding
  - Numeric scaling
  - New derived features
  - Train/test split
  - Saving preprocessors as joblib artifacts

Run:
    python feature_engineering.py
Outputs:
    data/X_train.pkl, data/X_test.pkl,
    data/y_train.pkl, data/y_test.pkl,
    models/preprocessor.joblib
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Feature Engineering  (domain-driven new features)
# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful new features without introducing data leakage.

    New features added
    ------------------
    avg_monthly_spend       : total_charges / tenure_months  (estimated average)
    charge_per_service      : monthly_charges / num_services
    num_services            : count of active add-on services
    is_high_value           : monthly_charges > 75th-percentile
    tenure_group            : categorical bucket of tenure
    ticket_total            : tech + admin tickets combined
    has_any_ticket          : binary flag — has raised ≥1 ticket
    charge_x_tenure         : interaction term
    low_satisfaction        : binary flag — satisfaction_score < 3
    """
    df = df.copy()

    # Number of active services
    service_cols = [
        "phone_service", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    ]
    # Count binary "1" / "Yes"
    def is_active(col_val):
        return (col_val == "Yes") | (col_val == 1)

    df["num_services"] = sum(
        is_active(df[c]).astype(int) for c in service_cols
    )

    # Average monthly spend
    df["avg_monthly_spend"] = (
        df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
        / df["tenure_months"].replace(0, np.nan)
    ).round(2)

    # Charge per service (avoid division by zero)
    df["charge_per_service"] = (
        df["monthly_charges"] / df["num_services"].replace(0, np.nan)
    ).fillna(df["monthly_charges"]).round(2)

    # High-value customer flag
    threshold = df["monthly_charges"].quantile(0.75)
    df["is_high_value"] = (df["monthly_charges"] > threshold).astype(int)

    # Tenure groups
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    df["tenure_group"] = pd.cut(
        df["tenure_months"], bins=bins, labels=labels, right=True
    ).astype(str)

    # Support ticket features
    df["ticket_total"] = df["num_tech_tickets"] + df["num_admin_tickets"]
    df["has_any_ticket"] = (df["ticket_total"] > 0).astype(int)

    # Interaction: high charge AND long tenure → usually loyal
    df["charge_x_tenure"] = (df["monthly_charges"] * df["tenure_months"]).round(2)

    # Low satisfaction flag
    df["low_satisfaction"] = (df["satisfaction_score"].fillna(3.5) < 3).astype(int)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build sklearn ColumnTransformer
# ──────────────────────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame):
    """
    Returns a fitted ColumnTransformer that:
      - imputes + scales numeric columns
      - imputes + one-hot-encodes categorical columns
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor, numeric_cols, categorical_cols


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_feature_engineering(
    input_path: str = "data/customer_churn.csv",
) -> tuple:
    """
    Full feature-engineering pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test  (as numpy arrays),
    preprocessor (fitted ColumnTransformer),
    feature_names (list[str])
    """
    print("\n🔧 Running Feature Engineering …\n")

    df = pd.read_csv(input_path)
    df = engineer_features(df)
    print(f"  Features after engineering: {df.shape[1]-2}")  # minus id & target

    # Drop identifiers; separate target
    drop_cols = ["customer_id", "churn"]
    X = df.drop(columns=drop_cols)
    y = df["churn"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  Train size : {len(X_train_raw):,}  |  Test size : {len(X_test_raw):,}")
    print(f"  Train churn rate : {y_train.mean():.2%}")
    print(f"  Test  churn rate : {y_test.mean():.2%}")

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Recover feature names (for SHAP, etc.)
    cat_feature_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    feature_names = num_cols + cat_feature_names

    print(f"  Total features after encoding : {len(feature_names)}")

    # Save artefacts
    joblib.dump(X_train, DATA_DIR / "X_train.pkl")
    joblib.dump(X_test, DATA_DIR / "X_test.pkl")
    joblib.dump(y_train, DATA_DIR / "y_train.pkl")
    joblib.dump(y_test, DATA_DIR / "y_test.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")

    # Also save raw split (needed for SHAP with tree models)
    X_train_raw.reset_index(drop=True).to_csv(DATA_DIR / "X_train_raw.csv")
    X_test_raw.reset_index(drop=True).to_csv(DATA_DIR / "X_test_raw.csv")
    pd.Series(y_train).to_csv(DATA_DIR / "y_train_raw.csv")
    pd.Series(y_test).to_csv(DATA_DIR / "y_test_raw.csv")

    print("\n✅  Saved preprocessor + train/test splits to data/ and models/")
    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    run_feature_engineering()
