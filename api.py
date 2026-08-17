"""
Customer Churn Prediction API

Flow:
    Raw customer JSON
        ↓
    Feature Engineering
        ↓
    Saved Preprocessor
        ↓
    Trained XGBoost Model
        ↓
    Churn Probability
        ↓
    Optimized Classification Threshold
        ↓
    Prediction + Risk Category
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction using the selected trained classification model",
    version="1.0.0",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"


MODEL_PATH = MODEL_DIR / "best_model_tuned.joblib"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

THRESHOLD_PATH = MODEL_DIR / "threshold.joblib"

FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None

preprocessor = None

threshold = 0.50

feature_names = None

MODEL_LOADED = False

PREPROCESSOR_LOADED = False

THRESHOLD_LOADED = False

MODEL_ERROR = None


# ============================================================
# LOAD FEATURE ENGINEERING FUNCTION
# ============================================================

try:

    from feature_engineering import engineer_features

    print("✅ Feature engineering function loaded")

except Exception as e:

    engineer_features = None

    print("❌ Could not import feature engineering")

    print(f"   Error: {e}")


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    MODEL_LOADED = True

    print("✅ Model loaded successfully")

    print(f"   Model: {MODEL_PATH}")


except Exception as e:

    MODEL_ERROR = str(e)

    print("❌ Model could not be loaded")

    print(f"   Error: {MODEL_ERROR}")


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

try:

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    PREPROCESSOR_LOADED = True

    print("✅ Preprocessor loaded successfully")

    print(f"   Preprocessor: {PREPROCESSOR_PATH}")


except Exception as e:

    print("❌ Preprocessor could not be loaded")

    print(f"   Error: {e}")


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

try:

    feature_names = joblib.load(FEATURE_NAMES_PATH)

    print("✅ Feature names loaded")

    print(f"   Number of features: {len(feature_names)}")


except Exception as e:

    feature_names = None

    print("⚠️ Feature names could not be loaded")

    print(f"   Error: {e}")


# ============================================================
# LOAD OPTIMIZED THRESHOLD
# ============================================================

try:

    threshold_data = joblib.load(THRESHOLD_PATH)

    # threshold.joblib contains a dictionary:
    #
    # {
    #     "threshold": 0.54,
    #     "optimization_metric": "f1",
    #     "source": "...",
    #     "metrics": {...}
    # }

    if isinstance(threshold_data, dict):

        threshold = float(
            threshold_data["threshold"]
        )

    else:

        threshold = float(threshold_data)

    THRESHOLD_LOADED = True

    print("✅ Optimized threshold loaded")

    print(
        f"   Classification threshold: {threshold:.4f}"
    )


except Exception as e:

    threshold = 0.50

    print("⚠️ Threshold could not be loaded")

    print(f"   Error: {e}")

    print("⚠️ Using default threshold: 0.50")


# ============================================================
# INPUT DATA SCHEMA
# ============================================================

class CustomerData(BaseModel):

    customer_id: str = "TEST_001"

    age: int = 34

    gender: str = "Male"

    senior_citizen: int = 0

    has_partner: int = 0

    has_dependents: int = 0

    tenure_months: int = 3

    contract_type: str = "Month-to-Month"

    phone_service: int = 1

    multiple_lines: str = "No"

    internet_service: str = "Fiber Optic"

    online_security: str = "No"

    online_backup: str = "No"

    device_protection: str = "No"

    tech_support: str = "No"

    streaming_tv: str = "Yes"

    streaming_movies: str = "Yes"

    paperless_billing: int = 1

    payment_method: str = "Electronic Check"

    monthly_charges: float = 95.5

    total_charges: float = 286.5

    data_usage_gb: float = 42.3

    num_tech_tickets: int = 4

    num_admin_tickets: int = 2

    satisfaction_score: float = 2.0

    @field_validator(
        "senior_citizen",
        "has_partner",
        "has_dependents",
        "phone_service",
        "paperless_billing",
        mode="before",
    )
    @classmethod
    def normalize_binary_fields(cls, value):
        """Accept 0/1 as well as common Yes/No representations."""
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)):
            if value in (0, 1):
                return int(value)
            raise ValueError("Binary fields must be 0/1 or Yes/No.")

        text = str(value).strip().lower()
        if text in {"yes", "y", "true", "1"}:
            return 1
        if text in {"no", "n", "false", "0"}:
            return 0

        raise ValueError(
            "Binary fields must be one of: Yes, No, 1, 0, True, False."
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "message": "Customer Churn Prediction API",

        "status": "running",

        "model_loaded": MODEL_LOADED,

        "preprocessor_loaded": PREPROCESSOR_LOADED,

        "threshold_loaded": THRESHOLD_LOADED,

        "threshold": round(threshold, 4),

    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_loaded": MODEL_LOADED,

        "preprocessor_loaded": PREPROCESSOR_LOADED,

        "threshold_loaded": THRESHOLD_LOADED,

        "threshold": round(threshold, 4),

        "model_error": MODEL_ERROR,

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model": str(MODEL_PATH),
        "model_type": type(model).__name__ if model is not None else None,

        "preprocessor": str(PREPROCESSOR_PATH),

        "threshold_file": str(THRESHOLD_PATH),

        "model_loaded": MODEL_LOADED,

        "preprocessor_loaded": PREPROCESSOR_LOADED,

        "threshold": round(threshold, 4),

        "number_of_features": (
            len(feature_names)
            if feature_names is not None
            else None
        ),

    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )


    # --------------------------------------------------------
    # Check preprocessor
    # --------------------------------------------------------

    if preprocessor is None:

        raise HTTPException(
            status_code=500,
            detail="Preprocessor is not loaded."
        )


    # --------------------------------------------------------
    # Check feature engineering
    # --------------------------------------------------------

    if engineer_features is None:

        raise HTTPException(
            status_code=500,
            detail="Feature engineering function is not available."
        )


    try:

        # ====================================================
        # STEP 1 — Convert Pydantic object to DataFrame
        # ====================================================

        customer_dict = customer.model_dump()

        customer_id = customer_dict.get(
            "customer_id",
            "UNKNOWN"
        )

        raw_df = pd.DataFrame(
            [customer_dict]
        )


        print("\n==============================")

        print("NEW PREDICTION")

        print("==============================")

        print(
            f"Customer ID: {customer_id}"
        )


        # ====================================================
        # STEP 2 — Feature Engineering
        # ====================================================

        engineered_df = engineer_features(
            raw_df
        )


        # ====================================================
        # STEP 3 — Remove columns not used by model
        # ====================================================

        X_raw = engineered_df.drop(
            columns=[
                "customer_id",
                "churn",
            ],
            errors="ignore"
        )


        # ====================================================
        # STEP 4 — Apply SAME preprocessor used during training
        # ====================================================

        X_processed = preprocessor.transform(
            X_raw
        )


        print(
            f"Processed feature shape: {X_processed.shape}"
        )


        # ====================================================
        # STEP 5 — Model Prediction Probability
        # ====================================================

        probability = model.predict_proba(
            X_processed
        )[0][1]


        # ====================================================
        # STEP 6 — Apply optimized threshold
        # ====================================================

        prediction = (

            1

            if probability >= threshold

            else 0

        )


        # ====================================================
        # STEP 7 — Risk Category
        # ====================================================

        if probability >= 0.70:

            risk_category = "Critical"

        elif probability >= 0.50:

            risk_category = "High"

        elif probability >= 0.30:

            risk_category = "Medium"

        else:

            risk_category = "Low"


        # ====================================================
        # STEP 8 — Recommended Action
        # ====================================================

        if risk_category == "Critical":

            recommended_action = (
                "Emergency retention call within 24 hours"
            )

        elif risk_category == "High":

            recommended_action = (
                "Contact customer and offer retention incentives"
            )

        elif risk_category == "Medium":

            recommended_action = (
                "Monitor customer and consider a loyalty offer"
            )

        else:

            recommended_action = (
                "Continue normal customer engagement"
            )


        # ====================================================
        # STEP 9 — Return API Response
        # ====================================================

        response = {

            "customer_id": customer_id,

            "churn_probability": round(
                probability * 100,
                2
            ),

            "prediction": (

                "Will Churn"

                if prediction == 1

                else "Will Stay"

            ),

            "prediction_value": prediction,

            "risk_category": risk_category,

            "classification_threshold": round(
                threshold,
                4
            ),

            "recommended_action": recommended_action,

            # Kept as a list as well so the Streamlit UI can display
            # one or more business actions consistently.
            "recommended_actions": [recommended_action],

        }


        print("\nPrediction result:")

        print(response)

        print("==============================\n")


        return response


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("\n❌ Prediction error:")

        print(str(e))

        raise HTTPException(

            status_code=400,

            detail=f"Prediction failed: {str(e)}"

        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )