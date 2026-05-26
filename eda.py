"""
eda.py
======
Exploratory Data Analysis for the Customer Churn dataset.
Produces publication-quality plots saved under reports/figures/.

Run:
    python eda.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3f5c",
    "axes.labelcolor": "#c9d1e0",
    "xtick.color": "#8a93b0",
    "ytick.color": "#8a93b0",
    "text.color": "#c9d1e0",
    "grid.color": "#2a2d3e",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

ACCENT = "#7c6af7"
POS_COLOR = "#e05c5c"   # churn = 1
NEG_COLOR = "#5ce0b8"   # churn = 0
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str = "data/customer_churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─── 1. Churn Distribution ────────────────────────────────────────────────────
def plot_churn_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Churn Distribution", fontsize=16, fontweight="bold", color="#e0e6f5")

    counts = df["churn"].value_counts()
    colors = [NEG_COLOR, POS_COLOR]
    axes[0].bar(["No Churn (0)", "Churn (1)"], counts.values, color=colors,
                edgecolor="#0f1117", linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 80, f"{v:,}\n({v/len(df):.1%})", ha="center",
                     fontsize=11, color="#e0e6f5")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Counts")

    axes[1].pie(counts.values, labels=["No Churn", "Churn"],
                colors=colors, autopct="%1.1f%%", startangle=90,
                textprops={"color": "#e0e6f5"},
                wedgeprops={"edgecolor": "#0f1117", "linewidth": 2})
    axes[1].set_title("Class Proportions")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 01_churn_distribution.png")


# ─── 2. Numeric Feature Distributions ────────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    num_cols = ["age", "tenure_months", "monthly_charges", "total_charges",
                "data_usage_gb", "satisfaction_score",
                "num_tech_tickets", "num_admin_tickets"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Numeric Feature Distributions by Churn", fontsize=15,
                 fontweight="bold", color="#e0e6f5")
    axes = axes.flatten()

    for ax, col in zip(axes, num_cols):
        for label, color in [(0, NEG_COLOR), (1, POS_COLOR)]:
            subset = df[df["churn"] == label][col].dropna()
            ax.hist(subset, bins=35, alpha=0.65, color=color,
                    label=f"Churn={label}", edgecolor="none", density=True)
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 02_numeric_distributions.png")


# ─── 3. Boxplots ─────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame) -> None:
    num_cols = ["age", "tenure_months", "monthly_charges",
                "total_charges", "data_usage_gb", "satisfaction_score"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Boxplots: Numeric Features vs Churn", fontsize=15,
                 fontweight="bold", color="#e0e6f5")
    axes = axes.flatten()

    for ax, col in zip(axes, num_cols):
        data_0 = df[df["churn"] == 0][col].dropna()
        data_1 = df[df["churn"] == 1][col].dropna()
        bp = ax.boxplot([data_0, data_1], patch_artist=True,
                        medianprops={"color": "#fff", "linewidth": 2},
                        whiskerprops={"color": "#8a93b0"},
                        capprops={"color": "#8a93b0"},
                        flierprops={"marker": ".", "markersize": 2,
                                    "markerfacecolor": "#8a93b080"})
        bp["boxes"][0].set_facecolor(NEG_COLOR + "99")
        bp["boxes"][1].set_facecolor(POS_COLOR + "99")
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_title(col.replace("_", " ").title())

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 03_boxplots.png")


# ─── 4. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=np.number).drop(columns=["customer_id"],
                                                       errors="ignore")
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap=sns.diverging_palette(240, 10, as_cmap=True),
                vmin=-1, vmax=1, linewidths=0.5, linecolor="#0f1117",
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold",
                 color="#e0e6f5", pad=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 04_correlation_heatmap.png")


# ─── 5. Categorical Features vs Churn ────────────────────────────────────────
def plot_categorical_features(df: pd.DataFrame) -> None:
    cat_cols = ["contract_type", "internet_service", "payment_method",
                "gender", "online_security", "tech_support"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Churn Rate by Categorical Features", fontsize=15,
                 fontweight="bold", color="#e0e6f5")
    axes = axes.flatten()

    for ax, col in zip(axes, cat_cols):
        churn_rate = (df.groupby(col)["churn"].mean() * 100).sort_values(ascending=True)
        colors = [POS_COLOR if v > 25 else ACCENT for v in churn_rate.values]
        bars = ax.barh(churn_rate.index, churn_rate.values, color=colors,
                       edgecolor="#0f1117", linewidth=1)
        for bar, val in zip(bars, churn_rate.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Churn Rate (%)")
        ax.axvline(df["churn"].mean() * 100, color="#fff", linestyle="--",
                   alpha=0.4, linewidth=1, label="Overall avg")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 05_categorical_churn_rates.png")


# ─── 6. Tenure & Monthly Charges scatter ─────────────────────────────────────
def plot_tenure_vs_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, zorder in [(0, NEG_COLOR, 1), (1, POS_COLOR, 2)]:
        sub = df[df["churn"] == label]
        ax.scatter(sub["tenure_months"], sub["monthly_charges"],
                   c=color, alpha=0.18 if label == 0 else 0.40,
                   s=6, label=f"Churn={label}", zorder=zorder)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Tenure vs Monthly Charges — Colored by Churn",
                 fontsize=14, fontweight="bold", color="#e0e6f5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_tenure_vs_charges.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 06_tenure_vs_charges.png")


# ─── 7. Support Tickets vs Churn ─────────────────────────────────────────────
def plot_tickets(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Support Tickets vs Churn", fontsize=14, fontweight="bold",
                 color="#e0e6f5")
    for ax, col in zip(axes, ["num_tech_tickets", "num_admin_tickets"]):
        pivot = df.groupby(col)["churn"].mean().reset_index()
        ax.bar(pivot[col], pivot["churn"] * 100, color=ACCENT, edgecolor="#0f1117")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title(col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_tickets_vs_churn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved 07_tickets_vs_churn.png")


def run_eda(path: str = "data/customer_churn.csv") -> None:
    print("\n📊 Running EDA …\n")
    df = load_data(path)

    # Quick summary
    print("\n── Missing values ──")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nChurn rate: {df['churn'].mean():.2%}")

    plot_churn_distribution(df)
    plot_numeric_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_categorical_features(df)
    plot_tenure_vs_charges(df)
    plot_tickets(df)

    print(f"\n✅  All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    run_eda()
