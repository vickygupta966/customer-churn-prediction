"""Choose a classification threshold using out-of-fold training predictions.

The threshold is optimized without looking at the final test set. The selected
threshold is saved for the API/UI to use consistently.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def select_threshold(model, X_train, y_train) -> tuple[float, dict]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probability = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = {}

    for threshold in thresholds:
        pred = (oof_probability >= threshold).astype(int)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_metrics = {
                "f1": round(float(f1), 4),
                "precision": round(float(precision_score(y_train, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_train, pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_train, oof_probability)), 4),
            }

    return best_threshold, best_metrics


def run_threshold_optimization() -> float:
    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    model = joblib.load(MODEL_DIR / "best_model_tuned.joblib")

    threshold, metrics = select_threshold(model, X_train, y_train)
    artifact = {
        "threshold": threshold,
        "optimization_metric": "F1",
        "source": "5-fold out-of-fold training predictions",
        "metrics": metrics,
    }
    joblib.dump(artifact, MODEL_DIR / "threshold.joblib")

    print(f"Selected threshold: {threshold:.2f}")
    print(f"OOF metrics: {metrics}")
    print("Saved: models/threshold.joblib")
    return threshold


if __name__ == "__main__":
    run_threshold_optimization()
