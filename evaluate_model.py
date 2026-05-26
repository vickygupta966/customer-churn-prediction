"""
evaluate_model.py
=================
Comprehensive evaluation of the tuned model:
  - Classification report
  - Confusion matrix
  - ROC-AUC & PR curve
  - Calibration curve
  - Threshold analysis

Run:
    python evaluate_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss,
    accuracy_score, f1_score,
)
from sklearn.calibration import calibration_curve

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
FIG_DIR   = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f5c", "axes.labelcolor": "#c9d1e0",
    "xtick.color": "#8a93b0", "ytick.color": "#8a93b0",
    "text.color": "#c9d1e0", "grid.color": "#2a2d3e",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})


def load_artifacts():
    model = joblib.load(MODEL_DIR / "best_model_tuned.joblib")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")
    return model, X_test, y_test


def full_evaluation(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "═" * 60)
    print("  MODEL EVALUATION REPORT")
    print("═" * 60)
    print(f"\n  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  Avg Prec  : {average_precision_score(y_test, y_proba):.4f}")
    print(f"  Brier Scr : {brier_score_loss(y_test, y_proba):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Churn", "Churn"],
                                 digits=4))
    return y_pred, y_proba


def plot_all(model, X_test, y_test, y_pred, y_proba):
    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    ACCENT  = "#7c6af7"
    ORANGE  = "#e09c5c"
    GREEN   = "#5ce0b8"
    RED     = "#e05c5c"

    # ── 1. Confusion Matrix ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix", color="#e0e6f5")
    ax1.set_facecolor("#1a1d27")

    # ── 2. ROC Curve ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.4f}")
    ax2.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax2.plot([0,1],[0,1], "w--", lw=1, alpha=0.4)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve", color="#e0e6f5"); ax2.legend()

    # ── 3. Precision-Recall Curve ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax3.plot(rec, prec, color=GREEN, lw=2, label=f"AP = {ap:.4f}")
    ax3.fill_between(rec, prec, alpha=0.12, color=GREEN)
    ax3.axhline(y_test.mean(), color="w", linestyle="--", alpha=0.4,
                label="Baseline")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve", color="#e0e6f5"); ax3.legend()

    # ── 4. Calibration Curve ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=12)
    ax4.plot(mean_pred, frac_pos, "o-", color=ORANGE, lw=2, label="Model")
    ax4.plot([0,1],[0,1], "w--", alpha=0.4, label="Perfect")
    ax4.set_xlabel("Mean Predicted Prob"); ax4.set_ylabel("Fraction Positive")
    ax4.set_title("Calibration Curve", color="#e0e6f5"); ax4.legend()

    # ── 5. Score Distribution ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_proba[y_test == 0], bins=50, alpha=0.65, color=GREEN,
             label="No Churn", density=True)
    ax5.hist(y_proba[y_test == 1], bins=50, alpha=0.65, color=RED,
             label="Churn", density=True)
    ax5.set_xlabel("Predicted Churn Probability")
    ax5.set_ylabel("Density")
    ax5.set_title("Score Distribution by Class", color="#e0e6f5"); ax5.legend()

    # ── 6. Threshold vs F1/Recall/Precision ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(0.1, 0.9, 80)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_test, pred, zero_division=0))
        from sklearn.metrics import precision_score, recall_score
        precs.append(precision_score(y_test, pred, zero_division=0))
        recs.append(recall_score(y_test, pred, zero_division=0))
    ax6.plot(thresholds, f1s,   color=ACCENT, lw=2, label="F1")
    ax6.plot(thresholds, precs, color=GREEN,  lw=2, label="Precision")
    ax6.plot(thresholds, recs,  color=RED,    lw=2, label="Recall")
    best_t = thresholds[np.argmax(f1s)]
    ax6.axvline(best_t, color="white", linestyle="--", alpha=0.5,
                label=f"Best F1 threshold={best_t:.2f}")
    ax6.set_xlabel("Threshold"); ax6.set_ylabel("Score")
    ax6.set_title("Threshold Analysis", color="#e0e6f5"); ax6.legend(fontsize=8)

    fig.suptitle("Comprehensive Model Evaluation — Tuned XGBoost",
                 fontsize=15, fontweight="bold", color="#e0e6f5", y=1.01)
    fig.savefig(FIG_DIR / "11_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 11_model_evaluation.png")

    return best_t


def run_evaluation():
    model, X_test, y_test = load_artifacts()
    y_pred, y_proba = full_evaluation(model, X_test, y_test)
    best_threshold = plot_all(model, X_test, y_test, y_pred, y_proba)
    print(f"\n  📌 Optimal classification threshold (F1): {best_threshold:.2f}")
    joblib.dump(best_threshold, MODEL_DIR / "best_threshold.joblib")
    print("     Saved → models/best_threshold.joblib\n")


if __name__ == "__main__":
    run_evaluation()
