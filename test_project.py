"""
tests/test_project.py
=====================
Unit tests for the churn prediction pipeline.

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Dataset generation tests
# ──────────────────────────────────────────────────────────────────────────────
class TestDatasetGeneration:
    """Tests for generate_dataset.py"""

    @pytest.fixture(scope="class")
    def small_df(self, tmp_path_factory):
        from generate_dataset import generate_dataset
        path = tmp_path_factory.mktemp("data") / "test.csv"
        df = generate_dataset(n=500, save_path=str(path))
        return df

    def test_shape(self, small_df):
        assert small_df.shape[0] == 500
        assert small_df.shape[1] >= 16  # at least 15 features + 1 target

    def test_churn_column_binary(self, small_df):
        assert set(small_df["churn"].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, small_df):
        rate = small_df["churn"].mean()
        assert 0.10 <= rate <= 0.45, f"Churn rate {rate:.2%} outside expected range"

    def test_no_duplicate_ids(self, small_df):
        assert small_df["customer_id"].is_unique

    def test_required_columns_present(self, small_df):
        required = ["tenure_months", "monthly_charges", "contract_type",
                    "churn", "satisfaction_score", "num_tech_tickets"]
        for col in required:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_age_range(self, small_df):
        assert small_df["age"].min() >= 18
        assert small_df["age"].max() <= 85

    def test_tenure_range(self, small_df):
        assert small_df["tenure_months"].min() >= 1
        assert small_df["tenure_months"].max() <= 72

    def test_charges_positive(self, small_df):
        assert (small_df["monthly_charges"] > 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ──────────────────────────────────────────────────────────────────────────────
class TestFeatureEngineering:
    """Tests for feature_engineering.py"""

    @pytest.fixture(scope="class")
    def raw_df(self, tmp_path_factory):
        from generate_dataset import generate_dataset
        path = tmp_path_factory.mktemp("data") / "test.csv"
        return generate_dataset(n=300, save_path=str(path))

    @pytest.fixture(scope="class")
    def engineered_df(self, raw_df):
        from feature_engineering import engineer_features
        return engineer_features(raw_df)

    def test_new_feature_num_services(self, engineered_df):
        assert "num_services" in engineered_df.columns
        assert (engineered_df["num_services"] >= 0).all()
        assert (engineered_df["num_services"] <= 7).all()

    def test_tenure_groups(self, engineered_df):
        assert "tenure_group" in engineered_df.columns
        valid_groups = {"0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"}
        assert set(engineered_df["tenure_group"].unique()).issubset(valid_groups)

    def test_ticket_total(self, engineered_df):
        expected = (engineered_df["num_tech_tickets"]
                    + engineered_df["num_admin_tickets"])
        pd.testing.assert_series_equal(
            engineered_df["ticket_total"], expected,
            check_names=False
        )

    def test_is_high_value_binary(self, engineered_df):
        assert set(engineered_df["is_high_value"].unique()).issubset({0, 1})

    def test_avg_monthly_spend_positive(self, engineered_df):
        valid = engineered_df["avg_monthly_spend"].dropna()
        assert (valid > 0).all()

    def test_low_satisfaction_binary(self, engineered_df):
        assert set(engineered_df["low_satisfaction"].unique()).issubset({0, 1})

    def test_no_churn_leakage(self, engineered_df):
        """Ensure engineered features don't include future churn info."""
        assert "churn" in engineered_df.columns  # original churn is fine
        leakage_keywords = ["churn_next", "will_churn", "future"]
        for col in engineered_df.columns:
            for kw in leakage_keywords:
                assert kw not in col.lower(), f"Possible leakage column: {col}"

    def test_preprocessor_output_shape(self, raw_df):
        from feature_engineering import engineer_features, build_preprocessor
        X = engineer_features(raw_df).drop(columns=["customer_id", "churn"],
                                            errors="ignore")
        preprocessor, _, _ = build_preprocessor(X)
        X_trans = preprocessor.fit_transform(X)
        assert X_trans.shape[0] == len(raw_df)
        assert X_trans.shape[1] > 10


# ──────────────────────────────────────────────────────────────────────────────
# Prediction function tests
# ──────────────────────────────────────────────────────────────────────────────
class TestPrediction:
    """Tests for predict.py — requires trained model artifacts."""

    DEMO_CUSTOMER = {
        "customer_id": "TEST_001",
        "age": 34, "gender": "Male", "senior_citizen": 0,
        "has_partner": 0, "has_dependents": 0, "tenure_months": 3,
        "contract_type": "Month-to-Month", "phone_service": 1,
        "multiple_lines": "No", "internet_service": "Fiber Optic",
        "online_security": "No", "online_backup": "No",
        "device_protection": "No", "tech_support": "No",
        "streaming_tv": "Yes", "streaming_movies": "Yes",
        "paperless_billing": 1, "payment_method": "Electronic Check",
        "monthly_charges": 95.5, "total_charges": 286.5,
        "data_usage_gb": 42.3, "num_tech_tickets": 4,
        "num_admin_tickets": 2, "satisfaction_score": 2.0,
    }

    LOW_RISK_CUSTOMER = {
        "customer_id": "TEST_002",
        "age": 52, "gender": "Female", "senior_citizen": 0,
        "has_partner": 1, "has_dependents": 1, "tenure_months": 60,
        "contract_type": "Two Year", "phone_service": 1,
        "multiple_lines": "Yes", "internet_service": "DSL",
        "online_security": "Yes", "online_backup": "Yes",
        "device_protection": "Yes", "tech_support": "Yes",
        "streaming_tv": "No", "streaming_movies": "No",
        "paperless_billing": 0, "payment_method": "Bank Transfer",
        "monthly_charges": 48.0, "total_charges": 2880.0,
        "data_usage_gb": 15.0, "num_tech_tickets": 0,
        "num_admin_tickets": 0, "satisfaction_score": 4.8,
    }

    @pytest.fixture(scope="class")
    def predictor(self):
        """Skip if model artifacts don't exist."""
        model_path = Path("models/best_model_tuned.joblib")
        if not model_path.exists():
            pytest.skip("Model artifacts not found — run full pipeline first.")
        from predict import ChurnPredictor
        return ChurnPredictor()

    def test_probability_in_range(self, predictor):
        result = predictor.predict(self.DEMO_CUSTOMER)
        assert 0.0 <= result.churn_probability <= 1.0

    def test_risk_category_valid(self, predictor):
        result = predictor.predict(self.DEMO_CUSTOMER)
        valid = {"Low", "Medium", "High", "Critical"}
        assert result.risk_category in valid

    def test_high_risk_customer(self, predictor):
        """Month-to-month, short tenure, many tickets → high churn prob."""
        result = predictor.predict(self.DEMO_CUSTOMER)
        assert result.churn_probability > 0.5, (
            f"Expected high churn prob, got {result.churn_probability:.2%}"
        )

    def test_low_risk_customer(self, predictor):
        """Two-year contract, long tenure, satisfied → low churn prob."""
        result = predictor.predict(self.LOW_RISK_CUSTOMER)
        assert result.churn_probability < 0.5, (
            f"Expected low churn prob, got {result.churn_probability:.2%}"
        )

    def test_top_risk_factors_not_empty(self, predictor):
        result = predictor.predict(self.DEMO_CUSTOMER)
        assert len(result.top_risk_factors) >= 1

    def test_recommended_actions_not_empty(self, predictor):
        result = predictor.predict(self.DEMO_CUSTOMER)
        assert len(result.recommended_actions) >= 1

    def test_to_dict(self, predictor):
        result = predictor.predict(self.DEMO_CUSTOMER)
        d = result.to_dict()
        assert "churn_probability" in d
        assert "risk_category" in d

    def test_ordering_high_risk_gt_low_risk(self, predictor):
        high = predictor.predict(self.DEMO_CUSTOMER).churn_probability
        low  = predictor.predict(self.LOW_RISK_CUSTOMER).churn_probability
        assert high > low, (
            f"High-risk ({high:.2%}) should exceed low-risk ({low:.2%})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Risk thresholds
# ──────────────────────────────────────────────────────────────────────────────
class TestRiskCategories:
    def test_category_assignment(self):
        from predict import _risk_category
        assert _risk_category(0.10) == "Low"
        assert _risk_category(0.40) == "Medium"
        assert _risk_category(0.65) == "High"
        assert _risk_category(0.90) == "Critical"

    def test_boundary_values(self):
        from predict import _risk_category
        assert _risk_category(0.00) == "Low"
        assert _risk_category(0.30) == "Medium"
        assert _risk_category(0.55) == "High"
        assert _risk_category(0.75) == "Critical"
