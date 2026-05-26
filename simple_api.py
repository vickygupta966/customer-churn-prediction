import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Customer Churn Prediction API")

# Load model
try:
    model = joblib.load('models/best_model.pkl')
    print("✅ Model loaded successfully")
except:
    try:
        model = joblib.load('best_churn_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model not found: {e}")
        model = None

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: str = "Month-to-month"
    payment_method: str = "Electronic check"
    tech_tickets: int = 0
    satisfaction_score: int = 3

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([customer.dict()])
    
    try:
        proba = model.predict_proba(df)[0][1]
        prediction = 1 if proba > 0.5 else 0
        
        return {
            "churn_probability": round(proba * 100, 2),
            "prediction": "Will Churn" if prediction == 1 else "Will Stay",
            "risk_category": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
