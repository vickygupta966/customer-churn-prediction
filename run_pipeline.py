"""
run_pipeline.py
===============
End-to-end pipeline runner. Runs all steps in order.

Usage:
    python run_pipeline.py             # full pipeline
    python run_pipeline.py --skip-eda  # skip EDA (faster)
    python run_pipeline.py --optuna    # use Optuna for tuning
"""

import sys
import time
import argparse


def banner(text: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--optuna",   action="store_true")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: Generate dataset ────────────────────────────────────────────
    banner("STEP 1 — Generate Synthetic Dataset")
    from generate_dataset import generate_dataset
    generate_dataset()

    # ── Step 2: EDA ─────────────────────────────────────────────────────────
    if not args.skip_eda:
        banner("STEP 2 — Exploratory Data Analysis")
        from eda import run_eda
        run_eda()
    else:
        print("\n⏭  EDA skipped (--skip-eda)")

    # ── Step 3: Feature engineering ─────────────────────────────────────────
    banner("STEP 3 — Feature Engineering")
    from feature_engineering import run_feature_engineering
    run_feature_engineering()

    # ── Step 4: Train all models ─────────────────────────────────────────────
    banner("STEP 4 — Train & Evaluate All Models")
    from train_models import train_all_models
    train_all_models()

    # ── Step 5: Hyperparameter tuning ────────────────────────────────────────
    banner("STEP 5 — Hyperparameter Tuning")
    from hyperparameter_tuning import run_tuning
    run_tuning(use_optuna=args.optuna)

    # ── Step 6: Final evaluation ─────────────────────────────────────────────
    banner("STEP 6 — Comprehensive Model Evaluation")
    from evaluate_model import run_evaluation
    run_evaluation()

    # ── Step 7: SHAP interpretability ────────────────────────────────────────
    banner("STEP 7 — SHAP Interpretability")
    from interpretability import run_shap_analysis
    run_shap_analysis()

    # ── Step 8: Demo prediction ───────────────────────────────────────────────
    banner("STEP 8 — Demo Prediction")
    from predict import ChurnPredictor, DEMO_CUSTOMERS
    predictor = ChurnPredictor()
    for customer in DEMO_CUSTOMERS:
        print(predictor.predict(customer))

    elapsed = round(time.time() - t_start, 1)
    banner(f"✅  Full pipeline complete in {elapsed}s")
    print("\n  Outputs:")
    print("    data/          → dataset + preprocessed splits")
    print("    models/        → all trained models")
    print("    reports/       → comparison CSV, SHAP drivers")
    print("    reports/figures → all visualisation PNGs\n")


if __name__ == "__main__":
    main()
