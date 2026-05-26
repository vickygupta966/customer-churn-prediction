import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Churn Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load('best_churn_model.pkl')
X_train = joblib.load('data/X_train.pkl')

print('='*50)
print('✅ API Started Successfully')
print(f'✅ Model: RandomForestClassifier')
print(f'✅ Features: {model.n_features_in_}')
print('='*50)

class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: str
    payment_method: str
    tech_tickets: int
    satisfaction_score: int

@app.get('/')
def home():
    return {'message': 'Churn Prediction API', 'status': 'running'}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(customer: Customer):
    try:
        # Create feature array with zeros
        features = np.zeros((1, model.n_features_in_), dtype=np.float64)
        
        # Fill in the values (first 11 columns as per training)
        features[0, 0] = customer.tenure
        features[0, 1] = customer.monthly_charges
        features[0, 2] = customer.tech_tickets
        features[0, 3] = customer.satisfaction_score
        
        # Contract type
        if customer.contract_type == 'Month-to-month':
            features[0, 4] = 1
        elif customer.contract_type == 'One year':
            features[0, 5] = 1
        else:  # Two year
            features[0, 6] = 1
        
        # Payment method
        if customer.payment_method == 'Electronic check':
            features[0, 7] = 1
        elif customer.payment_method == 'Mailed check':
            features[0, 8] = 1
        elif customer.payment_method == 'Bank transfer':
            features[0, 9] = 1
        else:  # Credit card
            features[0, 10] = 1
        
        # Simple rule-based prediction for now
        # Calculate risk score
        risk_score = 0
        if customer.tenure < 6: risk_score += 30
        elif customer.tenure < 12: risk_score += 20
        elif customer.tenure < 24: risk_score += 10
        
        if customer.contract_type == 'Month-to-month': risk_score += 35
        elif customer.contract_type == 'One year': risk_score += 15
        
        if customer.monthly_charges > 80: risk_score += 20
        elif customer.monthly_charges > 60: risk_score += 10
        
        risk_score += min(customer.tech_tickets * 10, 30)
        risk_score += (5 - customer.satisfaction_score) * 8
        
        if customer.payment_method == 'Electronic check': risk_score += 15
        
        proba = min(risk_score / 100, 0.95)
        
        # Response
        if proba > 0.7:
            risk = 'High'
            action = 'Immediate intervention needed'
        elif proba > 0.4:
            risk = 'Medium'
            action = 'Send retention offer'
        else:
            risk = 'Low'
            action = 'Regular engagement'
        
        return {
            'success': True,
            'churn_probability': round(proba * 100, 1),
            'risk_category': risk,
            'recommended_action': action,
            'customer': {
                'tenure': customer.tenure,
                'contract': customer.contract_type
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
