"""
Final evaluation of the production churn model.

Important:
- Model was selected/tuned using training data only.
- Threshold was selected using out-of-fold training predictions only.
- The final test set is used here ONCE for final evaluation.
- Test set is never used for model/threshold selection.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_final_model():
    print("\n" + "=" * 70)
    print("🎯 FINAL PRODUCTION MODEL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load untouched test data
    # ---------------------------------------------------------

    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    print("\n📦 Test Data")
    print("-" * 70)
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    # ---------------------------------------------------------
    # 2. Load final production model
    # ---------------------------------------------------------

    model = joblib.load(MODEL_DIR / "best_model_tuned.joblib")

    # ---------------------------------------------------------
    # 3. Load optimized threshold
    # ---------------------------------------------------------

    threshold_artifact = joblib.load(MODEL_DIR / "threshold.joblib")

    threshold = float(threshold_artifact["threshold"])

    print("\n🤖 Production Model")
    print("-" * 70)
    print(f"Model     : {type(model).__name__}")
    print(f"Threshold : {threshold:.2f}")

    # ---------------------------------------------------------
    # 4. Generate probability predictions
    # ---------------------------------------------------------

    y_proba = model.predict_proba(X_test)[:, 1]

    # IMPORTANT:
    # Use optimized production threshold instead of 0.50.
    y_pred = (y_proba >= threshold).astype(int)

    # ---------------------------------------------------------
    # 5. Calculate final metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # ---------------------------------------------------------
    # 6. Confusion Matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("📊 FINAL TEST RESULTS")
    print("=" * 70)

    print(f"\nThreshold : {threshold:.2f}")

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(cm)

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    # ---------------------------------------------------------
    # 8. Classification report
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 9. Compare threshold 0.50 vs optimized threshold
    # ---------------------------------------------------------

    y_pred_default = (y_proba >= 0.50).astype(int)

    default_f1 = f1_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    default_precision = precision_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    default_recall = recall_score(
        y_test,
        y_pred_default,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("⚖️ THRESHOLD COMPARISON")
    print("=" * 70)

    print("\nDefault threshold = 0.50")
    print(f"F1        : {default_f1:.4f}")
    print(f"Precision : {default_precision:.4f}")
    print(f"Recall    : {default_recall:.4f}")

    print(f"\nOptimized threshold = {threshold:.2f}")
    print(f"F1        : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    print(f"\nF1 difference: {f1 - default_f1:+.4f}")

    # ---------------------------------------------------------
    # 10. Save final metrics
    # ---------------------------------------------------------

    metrics = {
        "model": type(model).__name__,
        "threshold": threshold,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "default_threshold": 0.50,
        "default_f1": round(float(default_f1), 4),
        "default_precision": round(float(default_precision), 4),
        "default_recall": round(float(default_recall), 4),
        "evaluation_dataset": "untouched test set",
    }

    # Save JSON-like joblib artifact
    joblib.dump(
        metrics,
        MODEL_DIR / "final_test_metrics.joblib",
    )

    # Save CSV
    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        REPORT_DIR / "final_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("✅ FINAL EVALUATION COMPLETE")
    print("=" * 70)

    print("\nSaved files:")
    print("  models/final_test_metrics.joblib")
    print("  reports/final_test_metrics.csv")

    print("\n⚠️ Important:")
    print("The test set was NOT used for model selection.")
    print("The test set was NOT used for threshold selection.")

    return metrics


if __name__ == "__main__":
    evaluate_final_model()