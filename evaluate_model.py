"""Evaluate the tuned churn model on the untouched test set.

The classification threshold is loaded from models/threshold.joblib, where it
was selected using out-of-fold training predictions. This module never chooses
or overwrites the threshold using the test set.

Run:
    python evaluate_model.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model_tuned.joblib")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    threshold_artifact = joblib.load(MODEL_DIR / "threshold.joblib")
    if isinstance(threshold_artifact, dict):
        threshold = float(threshold_artifact["threshold"])
    else:
        threshold = float(threshold_artifact)

    return model, X_test, y_test, threshold


def full_evaluation(model, X_test, y_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n" + "═" * 60)
    print("  FINAL TEST-SET MODEL EVALUATION")
    print("═" * 60)
    print(f"\n  Classification threshold : {threshold:.2f}")
    print(f"  Accuracy                 : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC                  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Average Precision        : {average_precision_score(y_test, y_proba):.4f}")
    print(f"  Brier Score              : {brier_score_loss(y_test, y_proba):.4f}")
    print(f"  Churn Precision          : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Churn Recall             : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Churn F1                 : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\n  Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            digits=4,
            zero_division=0,
        )
    )
    return y_pred, y_proba


def plot_all(X_test, y_test, y_pred, y_proba, threshold):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # 1. Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax1, colorbar=False)
    ax1.set_title("Confusion Matrix")

    # 2. ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    # 3. Precision-recall curve
    ax3 = fig.add_subplot(gs[0, 2])
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax3.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax3.axhline(y_test.mean(), linestyle="--", label="Baseline")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve")
    ax3.legend()

    # 4. Calibration curve
    ax4 = fig.add_subplot(gs[1, 0])
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=12)
    ax4.plot(mean_pred, frac_pos, "o-", lw=2, label="Model")
    ax4.plot([0, 1], [0, 1], "--", label="Perfect")
    ax4.set_xlabel("Mean Predicted Probability")
    ax4.set_ylabel("Fraction Positive")
    ax4.set_title("Calibration Curve")
    ax4.legend()

    # 5. Score distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_proba[y_test == 0], bins=50, alpha=0.65, label="No Churn", density=True)
    ax5.hist(y_proba[y_test == 1], bins=50, alpha=0.65, label="Churn", density=True)
    ax5.axvline(threshold, linestyle="--", lw=2, label=f"Threshold = {threshold:.2f}")
    ax5.set_xlabel("Predicted Churn Probability")
    ax5.set_ylabel("Density")
    ax5.set_title("Score Distribution by Class")
    ax5.legend()

    # 6. Threshold analysis on the test set is descriptive only.
    # It does NOT select or save a new threshold from test data.
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(0.10, 0.90, 81)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, pred, zero_division=0))
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))

    ax6.plot(thresholds, f1s, lw=2, label="F1")
    ax6.plot(thresholds, precs, lw=2, label="Precision")
    ax6.plot(thresholds, recs, lw=2, label="Recall")
    ax6.axvline(
        threshold,
        linestyle="--",
        lw=2,
        label=f"Selected threshold = {threshold:.2f}",
    )
    ax6.set_xlabel("Threshold")
    ax6.set_ylabel("Score")
    ax6.set_title("Threshold Analysis (Descriptive)")
    ax6.legend(fontsize=8)

    fig.suptitle("Final Test Evaluation — Tuned XGBoost", fontsize=15, fontweight="bold")
    fig.savefig(FIG_DIR / "11_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved reports/figures/11_model_evaluation.png")


def run_evaluation():
    model, X_test, y_test, threshold = load_artifacts()
    y_pred, y_proba = full_evaluation(model, X_test, y_test, threshold)
    plot_all(X_test, y_test, y_pred, y_proba, threshold)
    print("\n  Threshold source: 5-fold out-of-fold training optimization")
    print("  Test set was not used to select or overwrite the threshold.")


if __name__ == "__main__":
    run_evaluation()
