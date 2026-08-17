"""
Optimize the classification threshold using out-of-fold training predictions.

Important:
- The final test set is NEVER used to select the threshold.
- Threshold selection is based only on 5-fold OOF predictions from X_train.
- The selected threshold is saved for consistent API/UI predictions.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_threshold(
    model,
    X_train,
    y_train,
) -> tuple[float, dict]:
    """
    Generate out-of-fold probabilities on training data and
    select the threshold that maximizes F1 score.
    """

    print("\n" + "=" * 70)
    print("🎯 THRESHOLD OPTIMIZATION")
    print("=" * 70)

    print("\nCV folds       : 5")
    print("Scoring        : F1")
    print("Data used      : Training data only")
    print("Test set used  : NO")

    started = time.time()

    # --------------------------------------------------------
    # Generate out-of-fold probabilities
    # --------------------------------------------------------

    print("\nGenerating out-of-fold predictions...")

    oof_probability = cross_val_predict(
        estimator=clone(model),
        X=X_train,
        y=y_train,
        cv=CV,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    elapsed = round(time.time() - started, 2)

    print(f"OOF prediction time: {elapsed:.2f} seconds")

    # --------------------------------------------------------
    # Calculate baseline at threshold 0.50
    # --------------------------------------------------------

    baseline_threshold = 0.50

    baseline_pred = (
        oof_probability >= baseline_threshold
    ).astype(int)

    baseline_f1 = f1_score(
        y_train,
        baseline_pred,
        zero_division=0,
    )

    baseline_precision = precision_score(
        y_train,
        baseline_pred,
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_train,
        baseline_pred,
        zero_division=0,
    )

    print("\nDefault threshold = 0.50")
    print(f"F1        : {baseline_f1:.4f}")
    print(f"Precision : {baseline_precision:.4f}")
    print(f"Recall    : {baseline_recall:.4f}")

    # --------------------------------------------------------
    # Search thresholds
    # --------------------------------------------------------

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = {}

    threshold_results = []

    for threshold in thresholds:

        predictions = (
            oof_probability >= threshold
        ).astype(int)

        f1 = f1_score(
            y_train,
            predictions,
            zero_division=0,
        )

        precision = precision_score(
            y_train,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_train,
            predictions,
            zero_division=0,
        )

        threshold_results.append(
            {
                "threshold": round(float(threshold), 2),
                "f1": round(float(f1), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
            }
        )

        # Select threshold with highest F1.
        #
        # If two thresholds have the same F1,
        # prefer the lower threshold because churn
        # detection is generally recall-sensitive.
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

            best_metrics = {
                "f1": round(float(f1), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "roc_auc": round(
                    float(
                        roc_auc_score(
                            y_train,
                            oof_probability,
                        )
                    ),
                    4,
                ),
            }

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    threshold_results_path = (
        Path("reports") / "threshold_results.csv"
    )

    threshold_results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import pandas as pd

    pd.DataFrame(
        threshold_results
    ).to_csv(
        threshold_results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("🏆 BEST THRESHOLD")
    print("=" * 70)

    print(f"\nSelected threshold : {best_threshold:.2f}")
    print(f"OOF ROC-AUC        : {best_metrics['roc_auc']:.4f}")
    print(f"OOF F1             : {best_metrics['f1']:.4f}")
    print(f"OOF Precision      : {best_metrics['precision']:.4f}")
    print(f"OOF Recall         : {best_metrics['recall']:.4f}")

    print(
        f"\nThreshold improvement:"
        f" {baseline_f1:.4f} → {best_metrics['f1']:.4f}"
    )

    return best_threshold, best_metrics


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_threshold_optimization() -> float:
    """
    Load the tuned model and training data,
    optimize the threshold using OOF predictions,
    and save the threshold artifact.
    """

    print("\n" + "=" * 70)
    print("🚀 CUSTOMER CHURN THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\n📂 Loading training data...")

    X_train = joblib.load(
        DATA_DIR / "X_train.pkl"
    )

    y_train = joblib.load(
        DATA_DIR / "y_train.pkl"
    )

    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")

    # --------------------------------------------------------
    # Load tuned model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR / "best_model_tuned.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Tuned model not found:\n"
            f"{model_path}\n\n"
            f"Run hyperparameter_tuning.py first."
        )

    print("\n🤖 Loading tuned model...")

    model = joblib.load(model_path)

    print(f"Model type : {type(model).__name__}")

    # --------------------------------------------------------
    # Optimize threshold
    # --------------------------------------------------------

    threshold, metrics = select_threshold(
        model,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Save threshold artifact
    # --------------------------------------------------------

    artifact = {
        "threshold": float(threshold),
        "optimization_metric": "F1",
        "cv_folds": 5,
        "random_state": SEED,
        "source": "5-fold out-of-fold training predictions",
        "test_set_used_for_selection": False,
        "model_file": "best_model_tuned.joblib",
        "model_type": type(model).__name__,
        "metrics": metrics,
    }

    threshold_path = (
        MODEL_DIR / "threshold.joblib"
    )

    joblib.dump(
        artifact,
        threshold_path,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(f"\nSelected threshold : {threshold:.2f}")
    print(f"Model              : {type(model).__name__}")
    print(f"OOF ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"OOF F1             : {metrics['f1']:.4f}")
    print(f"OOF Precision      : {metrics['precision']:.4f}")
    print(f"OOF Recall         : {metrics['recall']:.4f}")

    print("\n📁 Saved files:")
    print(f"  {threshold_path}")
    print("  reports/threshold_results.csv")

    print("\n⚠️ Final test set was NOT used to select this threshold.")

    return threshold


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_threshold_optimization()