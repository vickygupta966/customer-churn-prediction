"""
Train and compare customer churn classification models.

IMPORTANT:
- The test set is NEVER used for model selection.
- Model selection is based only on 5-fold cross-validation ROC-AUC.
- After selecting the best model, the untouched test set is used once
  for final evaluation/reporting.
- The selected model is saved as best_model.joblib.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load already-prepared train/test datasets.
    """

    print("\n📂 Loading training and test data...")

    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")

    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models() -> dict:
    """
    Return all candidate classification models.
    """

    models = {

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.5,
            class_weight="balanced",
            random_state=SEED,
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # Neural Network
        # ----------------------------------------------------
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        ),
    }

    return models


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a fitted model on the untouched test set.
    """

    y_pred = model.predict(X_test)

    # Probability of positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(
            accuracy_score(y_test, y_pred),
            4,
        ),

        "Precision": round(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "Recall": round(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "F1": round(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "ROC-AUC": round(
            roc_auc_score(
                y_test,
                y_proba,
            ),
            4,
        ),
    }

    return metrics


# ============================================================
# TRAIN AND COMPARE
# ============================================================

def train_all_models():
    """
    Train all candidate models.

    Model selection:
        5-fold CV ROC-AUC

    Test set:
        Used ONLY after model selection for reporting.
    """

    print("\n" + "=" * 70)
    print("🚀 CUSTOMER CHURN MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    # --------------------------------------------------------
    # Get candidate models
    # --------------------------------------------------------

    models = get_models()

    cv_results = []
    fitted_models = {}

    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for name, model in models.items():

        print(f"\n🔄 Training: {name}")

        started = time.time()

        # ----------------------------------------------------
        # Cross-validation
        # ----------------------------------------------------

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV,
            scoring="roc_auc",
            n_jobs=-1,
        )

        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # ----------------------------------------------------
        # Fit model on complete training set
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        elapsed = round(
            time.time() - started,
            2,
        )

        # ----------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        row = {
            "Model": name,

            "CV ROC-AUC Mean": round(
                cv_mean,
                4,
            ),

            "CV ROC-AUC Std": round(
                cv_std,
                4,
            ),

            "Accuracy": test_metrics["Accuracy"],

            "Precision": test_metrics["Precision"],

            "Recall": test_metrics["Recall"],

            "F1": test_metrics["F1"],

            "ROC-AUC": test_metrics["ROC-AUC"],

            "Train Time (s)": elapsed,
        }

        cv_results.append(row)

        fitted_models[name] = model

        # ----------------------------------------------------
        # Save individual model
        # ----------------------------------------------------

        filename = (
            name
            .replace(" ", "_")
            .lower()
            + ".joblib"
        )

        joblib.dump(
            model,
            MODEL_DIR / filename,
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"   CV ROC-AUC : "
            f"{cv_mean:.4f} ± {cv_std:.4f}"
        )

        print(
            f"   Test ROC-AUC: "
            f"{test_metrics['ROC-AUC']:.4f}"
        )

        print(
            f"   Test F1     : "
            f"{test_metrics['F1']:.4f}"
        )

        print(
            f"   Train Time  : "
            f"{elapsed:.2f}s"
        )


    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        cv_results
    )

    # Sort ONLY by CV ROC-AUC
    results_df = results_df.sort_values(
        by="CV ROC-AUC Mean",
        ascending=False,
    ).reset_index(drop=True)


    # ========================================================
    # RANK MODELS
    # ========================================================

    results_df.insert(
        0,
        "Rank",
        range(
            1,
            len(results_df) + 1,
        ),
    )


    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = (
        REPORT_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )


    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_name = results_df.iloc[0]["Model"]

    best_model = fitted_models[best_name]


    # ========================================================
    # SAVE SELECTED MODEL
    # ========================================================

    best_model_path = (
        MODEL_DIR /
        "best_model.joblib"
    )

    selected_name_path = (
        MODEL_DIR /
        "selected_model_name.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path,
    )

    joblib.dump(
        best_name,
        selected_name_path,
    )


    # ========================================================
    # SAVE MODEL SELECTION METADATA
    # ========================================================

    metadata = {
        "selected_model": best_name,
        "selection_metric": "CV ROC-AUC Mean",
        "cv_strategy": "5-Fold Stratified Cross Validation",
        "random_state": SEED,
        "test_set_used_for_selection": False,
        "test_set_used_for_reporting": True,
        "best_cv_roc_auc": float(
            results_df.iloc[0]["CV ROC-AUC Mean"]
        ),
        "best_test_roc_auc": float(
            results_df.iloc[0]["ROC-AUC"]
        ),
        "best_test_f1": float(
            results_df.iloc[0]["F1"]
        ),
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "model_selection_metadata.joblib",
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("🏆 FINAL MODEL SELECTION")
    print("=" * 70)

    print(
        f"\nSelected Model: {best_name}"
    )

    print(
        f"CV ROC-AUC: "
        f"{results_df.iloc[0]['CV ROC-AUC Mean']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{results_df.iloc[0]['ROC-AUC']:.4f}"
    )

    print(
        f"Test F1: "
        f"{results_df.iloc[0]['F1']:.4f}"
    )

    print(
        "\n📁 Saved:"
    )

    print(
        f"   {best_model_path}"
    )

    print(
        f"   {selected_name_path}"
    )

    print(
        f"   {comparison_path}"
    )

    print(
        f"   {MODEL_DIR / 'model_selection_metadata.joblib'}"
    )

    print("\n" + "=" * 70)

    print(
        "\n⚠️ IMPORTANT:"
    )

    print(
        "The test set was NOT used to select the model."
    )

    print(
        "Model selection was based only on 5-fold CV ROC-AUC."
    )

    print("=" * 70)


    return (
        fitted_models,
        results_df,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_all_models()