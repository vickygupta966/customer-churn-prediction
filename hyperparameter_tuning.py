"""
Hyperparameter tuning for the model selected by train_models.py.

IMPORTANT:
- The test set is NEVER used during hyperparameter tuning.
- The selected model is read from selected_model_name.joblib.
- RandomizedSearchCV uses only X_train and y_train.
- The untouched test set is evaluated only after tuning is complete.
- The final tuned model is saved as best_model_tuned.joblib.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load train/test datasets.
    """

    print("\n📂 Loading data...")

    X_train = joblib.load(
        DATA_DIR / "X_train.pkl"
    )

    X_test = joblib.load(
        DATA_DIR / "X_test.pkl"
    )

    y_train = joblib.load(
        DATA_DIR / "y_train.pkl"
    )

    y_test = joblib.load(
        DATA_DIR / "y_test.pkl"
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# GET SELECTED MODEL
# ============================================================

def get_selected_model_name() -> str:
    """
    Read the model selected by train_models.py.
    """

    path = (
        MODEL_DIR /
        "selected_model_name.joblib"
    )

    if not path.exists():

        raise FileNotFoundError(
            "\n❌ selected_model_name.joblib was not found.\n\n"
            "Run train_models.py first:\n"
            "python train_models.py\n"
        )

    model_name = joblib.load(path)

    print(
        f"\n🎯 Selected model: {model_name}"
    )

    return model_name


# ============================================================
# MODEL + PARAMETER SEARCH SPACE
# ============================================================

def get_model_and_parameters(
    model_name: str,
):
    """
    Return the base model and hyperparameter
    search space for the selected model.
    """

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    if model_name == "Logistic Regression":

        model = LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        )

        params = {
            "C": [
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
            ],

            "solver": [
                "liblinear",
                "lbfgs",
            ],

            "class_weight": [
                None,
                "balanced",
            ],

            "max_iter": [
                1000,
                2000,
                3000,
            ],
        }

        return model, params


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    elif model_name == "Random Forest":

        model = RandomForestClassifier(
            random_state=SEED,
            n_jobs=-1,
        )

        params = {

            "n_estimators": [
                200,
                300,
                400,
                500,
            ],

            "max_depth": [
                None,
                8,
                12,
                16,
                20,
            ],

            "min_samples_split": [
                2,
                5,
                10,
            ],

            "min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],

            "max_features": [
                "sqrt",
                "log2",
                None,
            ],

            "class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],
        }

        return model, params


    # ========================================================
    # XGBOOST
    # ========================================================

    elif model_name == "XGBoost":

        model = XGBClassifier(
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )

        params = {

            "n_estimators": [
                200,
                300,
                400,
                500,
            ],

            "max_depth": [
                3,
                4,
                5,
                6,
                8,
            ],

            "learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1,
                0.15,
            ],

            "subsample": [
                0.7,
                0.8,
                0.85,
                1.0,
            ],

            "colsample_bytree": [
                0.7,
                0.8,
                0.85,
                1.0,
            ],

            "scale_pos_weight": [
                1,
                1.5,
                2,
                2.5,
                3,
            ],

            "min_child_weight": [
                1,
                3,
                5,
            ],

            "gamma": [
                0,
                0.1,
                0.3,
            ],

            "reg_lambda": [
                1,
                2,
                5,
            ],
        }

        return model, params


    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    elif model_name == "Neural Network":

        model = MLPClassifier(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )

        params = {

            "hidden_layer_sizes": [
                (64,),
                (128,),
                (128, 64),
                (128, 64, 32),
                (64, 32),
            ],

            "activation": [
                "relu",
                "tanh",
            ],

            "solver": [
                "adam",
            ],

            "alpha": [
                0.0001,
                0.001,
                0.01,
                0.1,
            ],

            "learning_rate": [
                "constant",
                "adaptive",
            ],

            "learning_rate_init": [
                0.0001,
                0.001,
                0.01,
            ],

            "batch_size": [
                32,
                64,
                128,
            ],
        }

        return model, params


    # ========================================================
    # UNKNOWN MODEL
    # ========================================================

    else:

        raise ValueError(
            f"\n❌ Unknown model: {model_name}\n"
            f"Expected one of:\n"
            f"  - Logistic Regression\n"
            f"  - Random Forest\n"
            f"  - XGBoost\n"
            f"  - Neural Network\n"
        )


# ============================================================
# EVALUATE FINAL MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the tuned model on the untouched test set.
    """

    y_pred = model.predict(
        X_test
    )

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "Accuracy": round(
            accuracy_score(
                y_test,
                y_pred,
            ),
            4,
        ),

        "Precision": round(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "Recall": round(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "F1": round(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4,
        ),

        "ROC-AUC": round(
            roc_auc_score(
                y_test,
                y_proba,
            ),
            4,
        ),
    }

    return metrics


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_selected_model(
    X_train,
    y_train,
    model_name,
):
    """
    Tune the selected model using randomized 5-fold CV.

    ONLY X_train and y_train are used here.
    """

    print("\n" + "=" * 70)

    print(
        f"🔧 HYPERPARAMETER TUNING: {model_name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Get model and search space
    # --------------------------------------------------------

    base_model, param_distributions = (
        get_model_and_parameters(
            model_name
        )
    )

    # --------------------------------------------------------
    # Number of iterations
    # --------------------------------------------------------

    if model_name == "Logistic Regression":

        n_iter = 30

    elif model_name == "Random Forest":

        n_iter = 40

    elif model_name == "XGBoost":

        n_iter = 40

    elif model_name == "Neural Network":

        n_iter = 30

    else:

        n_iter = 30


    # --------------------------------------------------------
    # Randomized Search
    # --------------------------------------------------------

    print(
        f"\n🔍 RandomizedSearchCV"
    )

    print(
        f"Candidates: {n_iter}"
    )

    print(
        f"CV folds: 5"
    )

    print(
        f"Scoring: ROC-AUC"
    )

    print(
        "\n⚠️ Test set is NOT being used."
    )


    started = time.time()


    search = RandomizedSearchCV(

        estimator=base_model,

        param_distributions=param_distributions,

        n_iter=n_iter,

        scoring="roc_auc",

        cv=CV,

        n_jobs=-1,

        refit=True,

        random_state=SEED,

        verbose=1,

        return_train_score=True,
    )


    search.fit(
        X_train,
        y_train,
    )


    elapsed = round(
        time.time() - started,
        2,
    )


    # ========================================================
    # SAVE SEARCH RESULTS
    # ========================================================

    results = (
        pd.DataFrame(
            search.cv_results_
        )
        .sort_values(
            "rank_test_score"
        )
    )

    results.to_csv(
        REPORT_DIR /
        "tuning_results.csv",
        index=False,
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print("🏆 BEST TUNING RESULT")

    print("=" * 70)

    print(
        f"\nModel: {model_name}"
    )

    print(
        f"Best CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"\nBest Parameters:"
    )

    for key, value in (
        search.best_params_.items()
    ):

        print(
            f"   {key}: {value}"
        )

    print(
        f"\nTuning Time: {elapsed:.2f} seconds"
    )


    return (
        search.best_estimator_,
        search.best_score_,
        search.best_params_,
        elapsed,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_tuning():
    """
    Complete hyperparameter tuning pipeline.
    """

    print("\n")
    print("=" * 70)
    print("🚀 CUSTOMER CHURN HYPERPARAMETER TUNING")
    print("=" * 70)


    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()


    # ========================================================
    # READ SELECTED MODEL
    # ========================================================

    model_name = (
        get_selected_model_name()
    )


    # ========================================================
    # TUNE SELECTED MODEL
    # ========================================================

    (
        tuned_model,
        best_cv_score,
        best_params,
        tuning_time,
    ) = tune_selected_model(

        X_train,

        y_train,

        model_name,
    )


    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "🧪 FINAL TEST EVALUATION"
    )

    print("=" * 70)

    print(
        "\n⚠️ The test set was untouched during tuning."
    )

    test_metrics = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )


    # ========================================================
    # DISPLAY TEST METRICS
    # ========================================================

    for metric, value in (
        test_metrics.items()
    ):

        print(
            f"{metric}: {value:.4f}"
        )


    # ========================================================
    # SAVE TUNED MODEL
    # ========================================================

    tuned_model_path = (
        MODEL_DIR /
        "best_model_tuned.joblib"
    )

    joblib.dump(
        tuned_model,
        tuned_model_path,
    )


    # ========================================================
    # SAVE TUNING METADATA
    # ========================================================

    metadata = {

        "model_name": model_name,

        "selection_stage":
            "CV model comparison",

        "tuning_method":
            "RandomizedSearchCV",

        "cv_folds": 5,

        "scoring":
            "ROC-AUC",

        "random_state":
            SEED,

        "best_cv_roc_auc":
            float(best_cv_score),

        "test_set_used_during_tuning":
            False,

        "test_metrics":
            test_metrics,

        "best_params":
            best_params,

        "tuning_time_seconds":
            tuning_time,
    }


    metadata_path = (
        MODEL_DIR /
        "tuning_metadata.joblib"
    )

    joblib.dump(
        metadata,
        metadata_path,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "✅ HYPERPARAMETER TUNING COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nSelected model:"
        f" {model_name}"
    )

    print(
        f"Best CV ROC-AUC:"
        f" {best_cv_score:.4f}"
    )

    print(
        f"Final Test ROC-AUC:"
        f" {test_metrics['ROC-AUC']:.4f}"
    )

    print(
        f"Final Test F1:"
        f" {test_metrics['F1']:.4f}"
    )

    print(
        "\n📁 Saved files:"
    )

    print(
        f"   {tuned_model_path}"
    )

    print(
        f"   {metadata_path}"
    )

    print(
        f"   {REPORT_DIR / 'tuning_results.csv'}"
    )

    print("\n" + "=" * 70)


    return tuned_model


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_tuning()