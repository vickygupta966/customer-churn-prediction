"""Train and compare churn models without using the test set for model selection."""

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

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def get_models() -> dict:
    """Return candidate classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED, C=0.5, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
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


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a fitted model on the untouched test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_all_models() -> tuple[dict, pd.DataFrame]:
    """Select the model by CV ROC-AUC, then evaluate once on the test set."""
    print("\n🚀 Training and comparing models …\n")

    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    models = get_models()
    cv_results = []
    fitted_models = {}

    for name, model in models.items():
        started = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_train, y_train)
        elapsed = round(time.time() - started, 2)

        test_metrics = evaluate(model, X_test, y_test)
        row = {
            "Model": name,
            "CV ROC-AUC Mean": round(cv_scores.mean(), 4),
            "CV ROC-AUC Std": round(cv_scores.std(), 4),
            **test_metrics,
            "Train Time (s)": elapsed,
        }
        cv_results.append(row)
        fitted_models[name] = model
        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib")
        print(
            f"{name}: CV ROC-AUC={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
            f"Test ROC-AUC={test_metrics['ROC-AUC']:.4f}"
        )

    results_df = pd.DataFrame(cv_results).sort_values(
        "CV ROC-AUC Mean", ascending=False
    )
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["Model"]
    joblib.dump(fitted_models[best_name], MODEL_DIR / "best_model.joblib")
    joblib.dump(best_name, MODEL_DIR / "selected_model_name.joblib")

    print(f"\n🏆 Selected by 5-fold CV: {best_name}")
    print(results_df.to_string(index=False))
    print("\n⚠️ Test metrics above are reported for comparison only; the test set is not used for selection.")

    return fitted_models, results_df


if __name__ == "__main__":
    train_all_models()
