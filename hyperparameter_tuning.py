"""Tune XGBoost using training data only, then perform one final test evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    """Tune XGBoost with 5-fold CV on training data only."""
    base = XGBClassifier(
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )

    # Focused grid keeps the search practical while covering the main capacity,
    # learning-rate and regularization trade-offs.
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85],
        "colsample_bytree": [0.7, 0.85],
        "scale_pos_weight": [1, 2, 3],
    }

    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    pd.DataFrame(search.cv_results_).to_csv(
        REPORT_DIR / "tuning_results.csv", index=False
    )
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    return search.best_estimator_


def run_tuning() -> XGBClassifier:
    """Tune, save and evaluate the final candidate once on the test set."""
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    model = tune_xgboost(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    joblib.dump(model, MODEL_DIR / "best_model_tuned.joblib")
    print(f"Final untouched test ROC-AUC: {test_auc:.4f}")
    print("Saved: models/best_model_tuned.joblib")
    return model


if __name__ == "__main__":
    run_tuning()
