"""
interpretability.py
===================
SHAP analysis for the tuned XGBoost model.
Produces:
  - Global feature importance (bar + beeswarm)
  - Local explanation for a single prediction
  - Dependence plots for top-3 features

Run:
    python interpretability.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
FIG_DIR   = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f5c", "axes.labelcolor": "#c9d1e0",
    "xtick.color": "#8a93b0", "ytick.color": "#8a93b0",
    "text.color": "#c9d1e0",
})

# Max rows to compute SHAP on (speed vs completeness trade-off)
SHAP_SAMPLE = 2_000


def run_shap_analysis():
    print("\n🔍 Running SHAP interpretability analysis …\n")

    model = joblib.load(MODEL_DIR / "best_model_tuned.joblib")
    X_test = joblib.load(DATA_DIR / "X_test.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    # Sample for speed
    np.random.seed(42)
    idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample = X_test[idx]

    # Build explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # XGBoost binary returns (n, features) for class 1 directly
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── 1. Bar plot — mean |SHAP| ──────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    top_n = 20
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#7c6af7" if v > top_vals.mean() else "#5ce0b8" for v in top_vals]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1],
            edgecolor="#0f1117")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Features by Mean |SHAP|",
                 fontsize=13, fontweight="bold", color="#e0e6f5")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 12_shap_bar.png")

    # ── 2. Beeswarm / Summary plot ─────────────────────────────────────────
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.gcf().set_facecolor("#0f1117")
    plt.title("SHAP Beeswarm — Feature Impact on Churn Probability",
              color="#e0e6f5", fontsize=12)
    plt.savefig(FIG_DIR / "13_shap_beeswarm.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  Saved 13_shap_beeswarm.png")

    # ── 3. Dependence plots — top 3 features ──────────────────────────────
    top3 = [feature_names[i] for i in np.argsort(mean_abs)[-3:][::-1]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SHAP Dependence Plots — Top 3 Features",
                 fontsize=13, fontweight="bold", color="#e0e6f5")

    for ax, feat in zip(axes, top3):
        if feat not in feature_names:
            continue
        fi = feature_names.index(feat)
        ax.scatter(X_sample[:, fi], sv[:, fi],
                   c=sv[:, fi], cmap="RdBu_r", alpha=0.35, s=5)
        ax.set_xlabel(feat, color="#c9d1e0")
        ax.set_ylabel("SHAP value", color="#c9d1e0")
        ax.axhline(0, color="white", lw=0.8, alpha=0.4)
        ax.set_title(feat, color="#e0e6f5")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 14_shap_dependence.png")

    # ── 4. Single-prediction waterfall (high-risk customer) ───────────────
    churn_probs = model.predict_proba(X_sample)[:, 1]
    high_risk_i = np.argmax(churn_probs)

    exp = shap.Explanation(
        values=sv[high_risk_i],
        base_values=explainer.expected_value if not isinstance(
            explainer.expected_value, list) else explainer.expected_value[1],
        data=X_sample[high_risk_i],
        feature_names=feature_names,
    )
    shap.waterfall_plot(exp, max_display=15, show=False)
    plt.gcf().set_facecolor("#0f1117")
    plt.title("SHAP Waterfall — Highest-Risk Customer",
              color="#e0e6f5", fontsize=11)
    plt.savefig(FIG_DIR / "15_shap_waterfall.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  Saved 15_shap_waterfall.png")

    # ── 5. Print top churn drivers ────────────────────────────────────────
    driver_df = pd.DataFrame({
        "Feature":         [feature_names[i] for i in top_idx],
        "Mean |SHAP|":     top_vals.round(4),
        "Direction":       ["↑ Increases churn" if sv[:, i].mean() > 0
                            else "↓ Decreases churn" for i in top_idx],
    })
    print("\n  📊 Top Churn Drivers:\n")
    print(driver_df.to_string(index=False))
    driver_df.to_csv("reports/shap_top_drivers.csv", index=False)
    print("\n  💾 Saved → reports/shap_top_drivers.csv")
    print("\n✅  SHAP analysis complete.\n")


if __name__ == "__main__":
    run_shap_analysis()
