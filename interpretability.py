"""
interpretability.py
===================
SHAP-style interpretability for the tuned XGBoost model.

Uses XGBoost native pred_contribs=True to avoid
SHAP/XGBoost 3.x base_score compatibility issues.

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
import xgboost as xgb
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

SHAP_SAMPLE = 2000


def run_shap_analysis():

    print("\n🔍 Running SHAP interpretability analysis ...\n")

    # ---------------------------------------------------------
    # Load model and data
    # ---------------------------------------------------------

    model = joblib.load(
        MODEL_DIR / "best_model_tuned.joblib"
    )

    X_test = joblib.load(
        DATA_DIR / "X_test.pkl"
    )

    feature_names = joblib.load(
        MODEL_DIR / "feature_names.joblib"
    )

    # ---------------------------------------------------------
    # Sample test data
    # ---------------------------------------------------------

    np.random.seed(42)

    idx = np.random.choice(
        len(X_test),
        min(SHAP_SAMPLE, len(X_test)),
        replace=False
    )

    X_sample = X_test[idx]

    # ---------------------------------------------------------
    # XGBoost native SHAP contributions
    # ---------------------------------------------------------

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(X_sample)

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    # Last column is the base value.
    sv = contributions[:, :-1]
    base_values = contributions[:, -1]

    print(
        f"Computed native XGBoost contributions: "
        f"{sv.shape}"
    )

    # ---------------------------------------------------------
    # 1. Global feature importance
    # ---------------------------------------------------------

    mean_abs = np.abs(sv).mean(axis=0)

    top_n = min(20, len(feature_names))

    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    top_names = [
        feature_names[i]
        for i in top_idx
    ]

    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top_names[::-1],
        top_vals[::-1]
    )

    ax.set_xlabel("Mean |SHAP Contribution|")

    ax.set_title(
        f"Top {top_n} Features by Mean |SHAP|"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "12_shap_bar.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("  Saved 12_shap_bar.png")

    # ---------------------------------------------------------
    # 2. SHAP beeswarm
    # ---------------------------------------------------------

    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot"
    )

    plt.title(
        "SHAP Beeswarm — Feature Impact on Churn"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "13_shap_beeswarm.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("  Saved 13_shap_beeswarm.png")

    # ---------------------------------------------------------
    # 3. Dependence plots
    # ---------------------------------------------------------

    top3_idx = top_idx[:3]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    fig.suptitle(
        "SHAP Dependence Plots — Top 3 Features"
    )

    for ax, fi in zip(axes, top3_idx):

        ax.scatter(
            X_sample[:, fi],
            sv[:, fi],
            alpha=0.35,
            s=8
        )

        ax.axhline(
            0,
            linewidth=0.8
        )

        ax.set_xlabel(
            feature_names[fi]
        )

        ax.set_ylabel(
            "SHAP contribution"
        )

        ax.set_title(
            feature_names[fi]
        )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "14_shap_dependence.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("  Saved 14_shap_dependence.png")

    # ---------------------------------------------------------
    # 4. Highest-risk customer
    # ---------------------------------------------------------

    churn_probs = model.predict_proba(
        X_sample
    )[:, 1]

    high_risk_i = np.argmax(
        churn_probs
    )

    high_risk_probability = churn_probs[
        high_risk_i
    ]

    # ---------------------------------------------------------
    # Create SHAP Explanation
    # ---------------------------------------------------------

    explanation = shap.Explanation(
        values=sv[high_risk_i],
        base_values=base_values[high_risk_i],
        data=X_sample[high_risk_i],
        feature_names=feature_names
    )

    shap.waterfall_plot(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        f"SHAP Waterfall — Highest-Risk Customer "
        f"({high_risk_probability:.1%} churn probability)"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "15_shap_waterfall.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("  Saved 15_shap_waterfall.png")

    # ---------------------------------------------------------
    # 5. Top churn drivers
    # ---------------------------------------------------------

    driver_df = pd.DataFrame({
        "Feature": [
            feature_names[i]
            for i in top_idx
        ],

        "Mean |SHAP|": [
            round(mean_abs[i], 4)
            for i in top_idx
        ],

        "Direction": [
            "↑ Increases churn"
            if sv[:, i].mean() > 0
            else "↓ Decreases churn"
            for i in top_idx
        ]
    })

    print("\n📊 Top Churn Drivers:\n")

    print(
        driver_df.to_string(
            index=False
        )
    )

    driver_df.to_csv(
        "reports/shap_top_drivers.csv",
        index=False
    )

    print(
        "\n💾 Saved → "
        "reports/shap_top_drivers.csv"
    )

    print(
        "\n✅ SHAP analysis complete.\n"
    )


if __name__ == "__main__":
    run_shap_analysis()
