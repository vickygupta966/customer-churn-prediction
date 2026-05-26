"""
predict.py
==========
Production-ready prediction module.

Usage (Python API):
    from predict import ChurnPredictor
    predictor = ChurnPredictor()
    result = predictor.predict(customer_dict)
    print(result)

Usage (CLI):
    python predict.py --demo
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any


MODEL_DIR = Path("models")
DATA_DIR  = Path("data")

# Risk tier thresholds (can be calibrated to business cost matrix)
RISK_THRESHOLDS = {
    "Low":    (0.00, 0.30),
    "Medium": (0.30, 0.55),
    "High":   (0.55, 0.75),
    "Critical":(0.75, 1.01),
}


@dataclass
class ChurnPrediction:
    customer_id: str
    churn_probability: float
    risk_category: str
    top_risk_factors: list[str]
    recommended_actions: list[str]
    model_version: str = "xgb_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "─" * 52,
            f"  Customer:          {self.customer_id}",
            f"  Churn Probability: {self.churn_probability:.1%}",
            f"  Risk Category:     {self.risk_category}",
            "  Top Risk Factors:",
        ]
        for f in self.top_risk_factors:
            lines.append(f"    • {f}")
        lines.append("  Recommended Actions:")
        for a in self.recommended_actions:
            lines.append(f"    → {a}")
        lines.append("─" * 52)
        return "\n".join(lines)


def _risk_category(prob: float) -> str:
    for cat, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return cat
    return "Critical"


def _recommend(risk: str, top_factors: list[str]) -> list[str]:
    """Generate human-readable retention actions based on risk + drivers."""
    base = {
        "Low":     ["Send quarterly satisfaction survey",
                    "Offer loyalty rewards points"],
        "Medium":  ["Proactively reach out via email",
                    "Offer a service upgrade or bundle discount",
                    "Assign a dedicated account manager"],
        "High":    ["Immediate outreach via phone or chat",
                    "Offer contract lock-in with 15% discount",
                    "Escalate to retention team"],
        "Critical":["Emergency retention call within 24 hrs",
                    "Offer personalised win-back package",
                    "Flag for executive review"],
    }
    actions = base.get(risk, [])
    factor_actions = {
        "contract_type": "Upsell from Month-to-Month to Annual contract",
        "num_tech_tickets": "Dispatch technical support team immediately",
        "satisfaction_score": "Conduct satisfaction recovery call",
        "monthly_charges": "Review billing and offer applicable discounts",
        "internet_service": "Offer fibre upgrade with promotional pricing",
        "tech_support": "Enable free tech support add-on for 3 months",
        "online_security": "Activate free security suite trial",
    }
    for f in top_factors:
        key = f.split("_")[0] + "_" + f.split("_")[1] if f.count("_") >= 1 else f
        for k, v in factor_actions.items():
            if k in f and v not in actions:
                actions.append(v)
                break
    return actions[:5]


class ChurnPredictor:
    """
    Wraps the trained preprocessor + model for single-customer inference.

    Parameters
    ----------
    model_path : str
        Path to joblib model file.
    preprocessor_path : str
        Path to fitted ColumnTransformer.
    """

    def __init__(
        self,
        model_path: str = "models/best_model_tuned.joblib",
        preprocessor_path: str = "models/preprocessor.joblib",
        feature_names_path: str = "models/feature_names.joblib",
    ):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = joblib.load(feature_names_path)

        # Load SHAP explainer lazily
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as training."""
        from feature_engineering import engineer_features
        return engineer_features(df)

    def predict(self, customer_data: dict | pd.DataFrame) -> ChurnPrediction:
        """
        Predict churn for a single customer.

        Parameters
        ----------
        customer_data : dict or single-row DataFrame
            Raw customer attributes (same schema as training data).

        Returns
        -------
        ChurnPrediction
        """
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()

        cid = df.get("customer_id", pd.Series(["UNKNOWN"])).iloc[0]
        if "customer_id" in df.columns:
            df = df.drop(columns=["customer_id"])
        if "churn" in df.columns:
            df = df.drop(columns=["churn"])

        df_eng = self._engineer_features(df)
        X = self.preprocessor.transform(df_eng)

        prob = float(self.model.predict_proba(X)[0, 1])
        risk = _risk_category(prob)

        # SHAP for local explanation
        try:
            explainer = self._get_explainer()
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]
            top_indices = np.argsort(np.abs(sv))[-5:][::-1]
            top_factors = [self.feature_names[i] for i in top_indices
                           if i < len(self.feature_names)]
        except Exception:
            top_factors = ["contract_type", "tenure_months",
                           "monthly_charges", "num_tech_tickets",
                           "satisfaction_score"]

        actions = _recommend(risk, top_factors)

        return ChurnPrediction(
            customer_id=str(cid),
            churn_probability=round(prob, 4),
            risk_category=risk,
            top_risk_factors=top_factors,
            recommended_actions=actions,
        )

    def predict_batch(
        self, df: pd.DataFrame, include_shap: bool = False
    ) -> pd.DataFrame:
        """
        Batch prediction for a DataFrame of customers.

        Returns original DataFrame with appended columns:
            churn_probability, risk_category
        """
        out = df.copy()
        ids = out.pop("customer_id") if "customer_id" in out.columns else pd.Series(
            range(len(out)), name="customer_id"
        )
        if "churn" in out.columns:
            out.pop("churn")

        df_eng = self._engineer_features(out)
        X = self.preprocessor.transform(df_eng)
        proba = self.model.predict_proba(X)[:, 1]

        df["churn_probability"] = proba.round(4)
        df["risk_category"] = [_risk_category(p) for p in proba]
        df["customer_id"] = ids.values
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Demo / CLI
# ──────────────────────────────────────────────────────────────────────────────
DEMO_CUSTOMERS = [
    {   # High churn risk
        "customer_id": "DEMO_001",
        "age": 34,
        "gender": "Male",
        "senior_citizen": 0,
        "has_partner": 0,
        "has_dependents": 0,
        "tenure_months": 3,
        "contract_type": "Month-to-Month",
        "phone_service": 1,
        "multiple_lines": "No",
        "internet_service": "Fiber Optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "paperless_billing": 1,
        "payment_method": "Electronic Check",
        "monthly_charges": 95.5,
        "total_charges": 286.5,
        "data_usage_gb": 42.3,
        "num_tech_tickets": 4,
        "num_admin_tickets": 2,
        "satisfaction_score": 2.0,
    },
    {   # Low churn risk — loyal customer
        "customer_id": "DEMO_002",
        "age": 52,
        "gender": "Female",
        "senior_citizen": 0,
        "has_partner": 1,
        "has_dependents": 1,
        "tenure_months": 60,
        "contract_type": "Two Year",
        "phone_service": 1,
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "paperless_billing": 0,
        "payment_method": "Bank Transfer",
        "monthly_charges": 48.0,
        "total_charges": 2880.0,
        "data_usage_gb": 15.0,
        "num_tech_tickets": 0,
        "num_admin_tickets": 1,
        "satisfaction_score": 4.5,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions")
    parser.add_argument("--input", type=str, help="Path to JSON with customer data")
    args = parser.parse_args()

    predictor = ChurnPredictor()

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        result = predictor.predict(data)
        print(result)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n🎯 Running demo predictions …\n")
        for customer in DEMO_CUSTOMERS:
            result = predictor.predict(customer)
            print(result)


if __name__ == "__main__":
    main()
