"""Run the complete customer churn training pipeline."""

from __future__ import annotations

import argparse
import time


def banner(text: str, width: int = 64) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer churn ML pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA")
    args = parser.parse_args()
    started = time.time()

    banner("STEP 1 — Generate Dataset")
    from generate_dataset import generate_dataset
    generate_dataset()

    if args.skip_eda:
        print("\n⏭ EDA skipped")
    else:
        banner("STEP 2 — Exploratory Data Analysis")
        from eda import run_eda
        run_eda()

    banner("STEP 3 — Leakage-Safe Feature Engineering")
    from feature_engineering import run_feature_engineering
    run_feature_engineering()

    banner("STEP 4 — Cross-Validated Model Comparison")
    from train_models import train_all_models
    train_all_models()

    banner("STEP 5 — XGBoost Hyperparameter Tuning")
    from hyperparameter_tuning import run_tuning
    run_tuning()

    banner("STEP 6 — Optimize Classification Threshold")
    from optimize_threshold import run_threshold_optimization
    run_threshold_optimization()

    banner("STEP 7 — Final Evaluation")
    from evaluate_model import run_evaluation
    run_evaluation()

    banner("STEP 8 — SHAP Interpretability")
    from interpretability import run_shap_analysis
    run_shap_analysis()

    banner("STEP 9 — Demo Predictions")
    from predict import ChurnPredictor, DEMO_CUSTOMERS
    predictor = ChurnPredictor()
    for customer in DEMO_CUSTOMERS:
        print(predictor.predict(customer))

    elapsed = round(time.time() - started, 1)
    banner(f"Pipeline complete in {elapsed}s")
    print("Artifacts:")
    print("  data/                     train/test datasets")
    print("  models/preprocessor.joblib")
    print("  models/best_model_tuned.joblib")
    print("  models/threshold.joblib")
    print("  reports/                  evaluation outputs")


if __name__ == "__main__":
    main()
