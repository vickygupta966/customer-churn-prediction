"""Tune XGBoost with randomized 5-fold CV on training data only."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    """Tune XGBoost with 40 randomized candidates and 5-fold CV."""
    base = XGBClassifier(
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "scale_pos_weight": [1, 1.5, 2, 2.5, 3],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [1, 2, 5],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
        refit=True,
        random_state=SEED,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )
    results.to_csv(REPORT_DIR / "tuning_results.csv", index=False)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    return search.best_estimator_


def run_tuning() -> XGBClassifier:
    """Tune on training data, then evaluate the selected model once on test data."""
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
