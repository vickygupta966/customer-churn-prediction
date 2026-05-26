"""
train_models.py
===============
Trains four models:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. Neural Network (PyTorch-free MLPClassifier for portability)

Evaluates each on the held-out test set and saves:
  - models/<model_name>.joblib
  - reports/model_comparison.csv

Run:
    python train_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f5c", "axes.labelcolor": "#c9d1e0",
    "xtick.color": "#8a93b0", "ytick.color": "#8a93b0",
    "text.color": "#c9d1e0", "grid.color": "#2a2d3e",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED, C=0.5, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3,          # handles class imbalance
            eval_metric="logloss", use_label_encoder=False,
            random_state=SEED, n_jobs=-1
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            max_iter=300, early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dict of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }


def plot_confusion_matrices(models: dict, X_test, y_test) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", color="#e0e6f5")

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, color="#e0e6f5")
        ax.set_facecolor("#1a1d27")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 08_confusion_matrices.png")


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    from sklearn.metrics import roc_curve
    colors = ["#7c6af7", "#5ce0b8", "#e05c5c", "#f7c86a"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#1a1d27")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")

    for (name, model), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold",
                 color="#e0e6f5")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 09_roc_curves.png")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#7c6af7", "#5ce0b8", "#e05c5c", "#f7c86a"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=row["Model"],
                      color=colors[i], edgecolor="#0f1117", alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center",
                    fontsize=7.5, color="#c9d1e0")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13,
                 fontweight="bold", color="#e0e6f5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 10_model_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def train_all_models():
    print("\n🚀 Training models …\n")

    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}\n")

    models = get_models()
    results = []

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        metrics = evaluate(model, X_test, y_test)
        metrics["Model"] = name
        metrics["Train Time (s)"] = elapsed
        results.append(metrics)

        # Print classification report
        print(f"─── {name} (trained in {elapsed}s) ───")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=["No Churn", "Churn"]))

        # Save model
        joblib.dump(model, MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib")
        print(f"  💾 Saved → models/{name.replace(' ', '_').lower()}.joblib\n")

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv("reports/model_comparison.csv")
    print("\n📊 Model Comparison:\n")
    print(results_df.to_string())

    # Plots
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_model_comparison(results_df.reset_index())

    best_model_name = results_df["ROC-AUC"].idxmax()
    print(f"\n🏆 Best model by ROC-AUC: {best_model_name}"
          f"  ({results_df.loc[best_model_name, 'ROC-AUC']:.4f})")
    joblib.dump(models[best_model_name], MODEL_DIR / "best_model.joblib")
    print("  Saved → models/best_model.joblib")

    return models, results_df


if __name__ == "__main__":
    train_all_models()
