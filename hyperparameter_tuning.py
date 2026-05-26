"""
hyperparameter_tuning.py
========================
Tunes the best-performing model (XGBoost) using:
  - Option A: GridSearchCV  (default, reproducible)
  - Option B: Optuna        (Bayesian, faster for large search spaces)

Run:
    python hyperparameter_tuning.py            # GridSearchCV
    python hyperparameter_tuning.py --optuna   # Optuna

Output:
    models/best_model_tuned.joblib
    reports/tuning_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBClassifier

SEED = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORER = make_scorer(roc_auc_score, needs_proba=True)


# ──────────────────────────────────────────────────────────────────────────────
# Option A: GridSearchCV
# ──────────────────────────────────────────────────────────────────────────────
def tune_gridsearch(X_train, y_train) -> XGBClassifier:
    """Exhaustive grid search over a focused parameter grid."""
    print("\n🔍 GridSearchCV tuning …\n")

    param_grid = {
        "n_estimators":    [200, 300, 400],
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.03, 0.05, 0.10],
        "subsample":       [0.7, 0.8, 0.9],
        "colsample_bytree":[0.7, 0.8],
        "scale_pos_weight":[2, 3],
    }

    base = XGBClassifier(
        eval_metric="logloss", use_label_encoder=False,
        random_state=SEED, n_jobs=-1
    )

    gs = GridSearchCV(
        base, param_grid,
        scoring=SCORER, cv=CV,
        n_jobs=-1, verbose=1, refit=True
    )
    gs.fit(X_train, y_train)

    print(f"\n  Best ROC-AUC (CV): {gs.best_score_:.4f}")
    print(f"  Best params: {gs.best_params_}")

    # Save tuning results
    results = pd.DataFrame(gs.cv_results_)
    results.to_csv(REPORT_DIR / "tuning_results.csv", index=False)

    return gs.best_estimator_


# ──────────────────────────────────────────────────────────────────────────────
# Option B: Optuna (Bayesian optimisation)
# ──────────────────────────────────────────────────────────────────────────────
def tune_optuna(X_train, y_train, n_trials: int = 60) -> XGBClassifier:
    """Bayesian hyper-parameter search using Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️  Optuna not installed. Falling back to GridSearchCV.")
        return tune_gridsearch(X_train, y_train)

    print(f"\n🔬 Optuna tuning ({n_trials} trials) …\n")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        }
        model = XGBClassifier(
            **params, eval_metric="logloss", use_label_encoder=False,
            random_state=SEED, n_jobs=-1
        )
        scores = cross_val_score(model, X_train, y_train,
                                  cv=CV, scoring=SCORER, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best ROC-AUC (CV): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best_model = XGBClassifier(
        **study.best_params,
        eval_metric="logloss", use_label_encoder=False,
        random_state=SEED, n_jobs=-1
    )
    best_model.fit(X_train, y_train)

    # Save trial history
    df_trials = study.trials_dataframe()
    df_trials.to_csv(REPORT_DIR / "tuning_results.csv", index=False)

    return best_model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run_tuning(use_optuna: bool = False):
    print("\n⚙️  Hyperparameter Tuning …\n")

    X_train = joblib.load(DATA_DIR / "X_train.pkl")
    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    X_test  = joblib.load(DATA_DIR / "X_test.pkl")
    y_test  = joblib.load(DATA_DIR / "y_test.pkl")

    if use_optuna:
        best_model = tune_optuna(X_train, y_train)
    else:
        best_model = tune_gridsearch(X_train, y_train)

    # Final evaluation on test set
    y_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"\n  ✅  Tuned model Test ROC-AUC : {test_auc:.4f}")

    # 5-fold CV on full training set
    cv_scores = cross_val_score(best_model, X_train, y_train,
                                 cv=CV, scoring=SCORER, n_jobs=-1)
    print(f"  CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(best_model, MODEL_DIR / "best_model_tuned.joblib")
    print("\n  💾 Saved → models/best_model_tuned.joblib")
    return best_model


if __name__ == "__main__":
    use_optuna = "--optuna" in sys.argv
    run_tuning(use_optuna=use_optuna)
