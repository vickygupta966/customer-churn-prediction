# 🔮 Customer Churn Prediction — End-to-End ML Project

> A production-grade machine learning pipeline that predicts which telecom customers are likely to churn, explains *why*, and recommends retention actions.

---

## 📌 Business Value

| Metric | Impact |
|--------|--------|
| **Churn rate reduction** | Targeting top 20% high-risk customers can cut overall churn by ~35% |
| **Revenue saved** | At ₹800/month ARPU and 5% churn on 100k customers → retaining 35% = **₹1.4 Cr/month** |
| **ROI on model** | Acquisition cost is 5–7× retention cost; ML-driven retention pays for itself in <1 month |

---

## 🏗️ Project Structure

```
churn_prediction/
├── generate_dataset.py       # Synthetic 20k-row dataset generator
├── eda.py                    # Exploratory data analysis + figures
├── feature_engineering.py   # Feature engineering + sklearn preprocessor
├── train_models.py           # Train LR, RF, XGBoost, Neural Net
├── hyperparameter_tuning.py  # GridSearchCV / Optuna tuning
├── evaluate_model.py         # ROC, PR curve, calibration, threshold analysis
├── interpretability.py       # SHAP global + local explanations
├── predict.py                # Production prediction function + CLI demo
├── api.py                    # FastAPI REST API
├── run_pipeline.py           # End-to-end pipeline runner
├── requirements.txt          # All dependencies with pinned versions
├── tests/
│   └── test_project.py       # Unit + integration tests (pytest)
├── data/                     # Generated after running pipeline
├── models/                   # Saved model artifacts
└── reports/
    ├── figures/               # All visualisation PNGs
    ├── model_comparison.csv   # Cross-model metric comparison
    └── shap_top_drivers.csv   # Top churn feature drivers
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
# Full pipeline (generates data → trains → evaluates → SHAP)
python run_pipeline.py

# Skip EDA for faster execution
python run_pipeline.py --skip-eda

# Use Optuna for Bayesian hyperparameter search
python run_pipeline.py --optuna
```

### 3. Run individual steps

```bash
python generate_dataset.py       # Step 1: create data/customer_churn.csv
python eda.py                    # Step 2: save plots to reports/figures/
python feature_engineering.py   # Step 3: preprocess + save train/test splits
python train_models.py           # Step 4: train 4 models, print metrics
python hyperparameter_tuning.py  # Step 5: tune best model
python evaluate_model.py         # Step 6: comprehensive evaluation
python interpretability.py       # Step 7: SHAP analysis
python predict.py --demo         # Step 8: run demo predictions
```

### 4. Launch API

```bash
uvicorn api:app --reload --port 8000
# Swagger docs → http://localhost:8000/docs
```

### 5. Run tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 📊 Dataset

**20,000 synthetic telecom customers** with 25 columns:

| Category | Features |
|----------|----------|
| Demographic | age, gender, senior_citizen, has_partner, has_dependents |
| Contract | tenure_months, contract_type |
| Services | phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies |
| Billing | paperless_billing, payment_method, monthly_charges, total_charges |
| Usage | data_usage_gb |
| Support | num_tech_tickets, num_admin_tickets |
| Satisfaction | satisfaction_score |
| **Target** | **churn** (0/1) |

Realistic churn rate: ~26% (industry average: 20–30%)

---

## 🔧 Feature Engineering

New derived features created without data leakage:

| Feature | Description |
|---------|-------------|
| `num_services` | Count of active add-on services (0–7) |
| `avg_monthly_spend` | total_charges / tenure_months |
| `charge_per_service` | monthly_charges / num_services |
| `is_high_value` | 1 if monthly_charges > 75th percentile |
| `tenure_group` | Bucketed tenure: 0-12, 13-24, 25-48, 49-72 months |
| `ticket_total` | tech_tickets + admin_tickets |
| `has_any_ticket` | Binary flag |
| `charge_x_tenure` | Interaction term |
| `low_satisfaction` | 1 if satisfaction_score < 3 |

---

## 🤖 Models & Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.80 | ~0.64 | ~0.71 | ~0.67 | ~0.87 |
| Random Forest | ~0.83 | ~0.68 | ~0.75 | ~0.71 | ~0.90 |
| **XGBoost (tuned)** | **~0.85** | **~0.72** | **~0.78** | **~0.75** | **~0.92** |
| Neural Network | ~0.82 | ~0.66 | ~0.73 | ~0.69 | ~0.89 |

*Results are approximate — exact values depend on random state and tuning.*

---

## 🔍 Interpretability (SHAP)

Top churn drivers identified by SHAP analysis:

1. **Contract type** — Month-to-Month customers churn 3× more
2. **Tenure** — First 12 months are highest risk
3. **Tech tickets** — 3+ tickets strongly predict churn
4. **Monthly charges** — Higher bills = higher churn risk
5. **Satisfaction score** — Score < 3 is a major red flag
6. **Internet service** — Fiber Optic has surprisingly higher churn
7. **Online security** — Lack of security add-on correlates with churn

---

## 🎯 Prediction API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
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
    "num_admin_tickets": 2, "satisfaction_score": 2.0
  }'
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.8741,
  "risk_category": "Critical",
  "top_risk_factors": ["contract_type_Month-to-Month", "num_tech_tickets", "satisfaction_score", "tenure_months", "monthly_charges"],
  "recommended_actions": ["Emergency retention call within 24 hrs", "Offer personalised win-back package", "Upsell from Month-to-Month to Annual contract", "Dispatch technical support team immediately", "Conduct satisfaction recovery call"],
  "model_version": "xgb_v1"
}
```

---

## ☁️ Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### AWS (Recommended path)

```
EC2 t3.medium  →  Docker  →  FastAPI  →  Application Load Balancer
                                              ↑
                                    Route 53 (custom domain)
S3 bucket ← model artifacts (versioned)
RDS / DynamoDB ← customer data + prediction logs
CloudWatch ← latency, error rate, drift monitoring
```

### GCP / Azure alternatives

- **GCP**: Cloud Run (serverless) + Vertex AI Model Registry
- **Azure**: Azure Container Instances + Azure ML

---

## 📈 MLOps Considerations

- **Model monitoring**: Track prediction score distributions weekly; alert if mean churn probability shifts >5%
- **Retraining trigger**: Retrain monthly or when ROC-AUC on labelled production data drops below 0.85
- **A/B testing**: Shadow-deploy new models, compare retention rates between control/treatment groups
- **Feature store**: Serve real-time features (recent tickets, last login) via Redis / Feast
- **Versioning**: MLflow / DVC for experiment tracking and model registry

---

## 🎤 Interview Presentation Guide

### Opening (30 seconds)
> "This project solves a ₹multi-crore business problem: predicting which customers will cancel their subscription before they do, so the retention team can intervene. I built an end-to-end ML pipeline from data generation through to a REST API, with SHAP explanations that a product manager can actually act on."

### Technical depth questions to prepare:
- *"Why XGBoost over Random Forest?"* → Better handling of missing values, scale_pos_weight for imbalance, faster inference
- *"How did you handle class imbalance?"* → scale_pos_weight in XGBoost + stratified splits; could also use SMOTE
- *"How would you detect model drift?"* → Monitor PSI (Population Stability Index) on input feature distributions
- *"What's your threshold strategy?"* → Business-driven: optimise for F1 or for precision/recall based on cost of false positives vs false negatives
- *"How would you deploy this at scale?"* → Docker + Kubernetes, async batch scoring for 10M customers overnight, real-time API for CRM integration

---

## 📋 Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for full SHAP)
- SHAP analysis takes ~2–3 minutes on CPU

---

## 📄 License

MIT — free to use, modify, and deploy.
