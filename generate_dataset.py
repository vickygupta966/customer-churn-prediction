"""
generate_dataset.py
====================
Generates a realistic 20,000-row telco customer churn dataset.
No data leakage — churn label is derived from probabilistic rules,
not from future-looking aggregates.

Run:
    python generate_dataset.py
Output:
    data/customer_churn.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 20_000
np.random.seed(SEED)


def generate_dataset(n: int = N, save_path: str = "data/customer_churn.csv") -> pd.DataFrame:
    """
    Generate a synthetic telco customer churn dataset.

    Parameters
    ----------
    n : int
        Number of rows.
    save_path : str
        File path where CSV will be saved.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = np.clip(np.random.normal(42, 14, n).astype(int), 18, 85)
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = (age >= 60).astype(int)
    has_partner = np.random.choice([0, 1], n, p=[0.48, 0.52])
    has_dependents = np.where(
        has_partner == 1,
        np.random.choice([0, 1], n, p=[0.55, 0.45]),
        np.random.choice([0, 1], n, p=[0.82, 0.18]),
    )

    # ── Tenure & Contract ─────────────────────────────────────────────────────
    # Tenure in months (1-72); new customers churn more
    tenure = np.clip(np.random.exponential(scale=30, size=n).astype(int), 1, 72)
    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n,
        p=[0.55, 0.25, 0.20],
    )

    # ── Service Features ──────────────────────────────────────────────────────
    internet_service = np.random.choice(
        ["DSL", "Fiber Optic", "No Internet"], n, p=[0.35, 0.45, 0.20]
    )
    phone_service = np.random.choice([0, 1], n, p=[0.10, 0.90])
    multiple_lines = np.where(
        phone_service == 1,
        np.random.choice(["Yes", "No"], n, p=[0.45, 0.55]),
        "No Phone",
    )
    online_security = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.38, 0.62]),
        "No Internet",
    )
    online_backup = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "No Internet",
    )
    device_protection = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.34, 0.66]),
        "No Internet",
    )
    tech_support = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.33, 0.67]),
        "No Internet",
    )
    streaming_tv = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.40, 0.60]),
        "No Internet",
    )
    streaming_movies = np.where(
        internet_service != "No Internet",
        np.random.choice(["Yes", "No"], n, p=[0.39, 0.61]),
        "No Internet",
    )

    # ── Billing & Payment ─────────────────────────────────────────────────────
    paperless_billing = np.random.choice([0, 1], n, p=[0.41, 0.59])
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges — driven by services
    base_charge = 20.0
    monthly_charges = (
        base_charge
        + (internet_service == "Fiber Optic") * np.random.uniform(30, 50, n)
        + (internet_service == "DSL") * np.random.uniform(10, 25, n)
        + (phone_service == 1) * np.random.uniform(5, 15, n)
        + (multiple_lines == "Yes") * np.random.uniform(5, 10, n)
        + (online_security == "Yes") * np.random.uniform(3, 7, n)
        + (online_backup == "Yes") * np.random.uniform(3, 7, n)
        + (device_protection == "Yes") * np.random.uniform(3, 7, n)
        + (tech_support == "Yes") * np.random.uniform(3, 7, n)
        + (streaming_tv == "Yes") * np.random.uniform(5, 10, n)
        + (streaming_movies == "Yes") * np.random.uniform(5, 10, n)
        + np.random.normal(0, 3, n)
    ).round(2)
    monthly_charges = np.clip(monthly_charges, 18, 120)

    # Total charges = monthly × tenure + small noise (realistic billing history)
    total_charges = (monthly_charges * tenure + np.random.normal(0, 20, n)).round(2)
    total_charges = np.clip(total_charges, monthly_charges, monthly_charges * 72)

    # ── Support Tickets ───────────────────────────────────────────────────────
    num_tech_tickets = np.random.poisson(lam=1.2, size=n)
    num_admin_tickets = np.random.poisson(lam=0.8, size=n)

    # ── Usage ─────────────────────────────────────────────────────────────────
    # GB of data used per month
    data_usage_gb = np.where(
        internet_service == "No Internet",
        0.0,
        np.clip(np.random.exponential(scale=25, size=n), 0.5, 200),
    ).round(2)

    # ── Satisfaction (latent; derived, not future-leaking) ────────────────────
    # This is a *current* survey score — collected at same time as other features
    satisfaction_score = np.clip(
        np.random.normal(3.5, 1.0, n)
        - 0.4 * (num_tech_tickets > 2).astype(float)
        - 0.3 * (internet_service == "Fiber Optic").astype(float)
        + 0.5 * (contract_type == "Two Year").astype(float),
        1,
        5,
    ).round(1)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{str(i).zfill(6)}" for i in range(1, n + 1)],
            "age": age,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "has_partner": has_partner,
            "has_dependents": has_dependents,
            "tenure_months": tenure,
            "contract_type": contract_type,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "data_usage_gb": data_usage_gb,
            "num_tech_tickets": num_tech_tickets,
            "num_admin_tickets": num_admin_tickets,
            "satisfaction_score": satisfaction_score,
        }
    )

    # ── Churn Label (probabilistic — no leakage) ──────────────────────────────
    # Logit score from realistic business drivers
    logit = (
        -2.5
        + 1.5 * (df["contract_type"] == "Month-to-Month").astype(float)
        - 0.04 * df["tenure_months"]
        + 0.015 * df["monthly_charges"]
        + 0.6 * (df["internet_service"] == "Fiber Optic").astype(float)
        - 0.4 * (df["tech_support"] == "Yes").astype(float)
        - 0.3 * (df["online_security"] == "Yes").astype(float)
        + 0.3 * (df["payment_method"] == "Electronic Check").astype(float)
        + 0.25 * df["num_tech_tickets"]
        - 0.5 * (df["satisfaction_score"] - 3)
        + 0.5 * df["senior_citizen"]
        + np.random.normal(0, 0.4, n)  # irreducible noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    df["churn"] = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # ── Inject realistic missingness ──────────────────────────────────────────
    for col, frac in [("total_charges", 0.005), ("satisfaction_score", 0.02)]:
        mask = np.random.choice([True, False], n, p=[frac, 1 - frac])
        df.loc[mask, col] = np.nan

    df.to_csv(save_path, index=False)
    print(f"✅  Dataset saved → {save_path}")
    print(f"    Shape : {df.shape}")
    print(f"    Churn rate : {df['churn'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_dataset()
