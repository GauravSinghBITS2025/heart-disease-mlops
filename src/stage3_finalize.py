"""
Stage 3 - MLflow refinement, reproducibility and model packaging.

This script:

1. Loads the Stage 2 final model.
2. Reconstructs the exact deterministic train/test split.
3. Saves the held-out test set.
4. Independently evaluates the final model.
5. Creates a reproducibility manifest.
6. Captures the Python/package environment.
7. Creates a dedicated MLflow final-model run.
8. Logs the model, parameters, metrics and artifacts.
"""

from pathlib import Path
import json
import platform
import sys
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from mlflow.models import infer_signature


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "final_model.joblib"
)

MODEL_METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "model_metadata.json"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

TEST_SET_FILE = REPORTS_DIR / "test_set.csv"
METRICS_FILE = REPORTS_DIR / "stage3_test_metrics.json"
MANIFEST_FILE = MODELS_DIR / "model_manifest.json"
ENVIRONMENT_FILE = REPORTS_DIR / "environment.txt"


# ============================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "target"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    """Create required Stage 3 directories."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load and validate the cleaned dataset."""

    print("Loading cleaned dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    print(f"Dataset shape: {df.shape}")

    return df


def load_final_model():
    """Load the Stage 2 final model."""

    print("Loading final model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(f"Model loaded from: {MODEL_FILE}")

    return model


def recreate_test_split(df):
    """
    Recreate the exact deterministic train/test split used in Stage 2.

    This works because Stage 2 used:

        test_size=0.20
        random_state=42
        stratify=y
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print("Recreated train/test split:")
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def save_test_set(X_test, y_test):
    """Save the exact held-out test set."""

    test_df = X_test.copy()

    test_df[TARGET_COLUMN] = y_test.values

    test_df.to_csv(TEST_SET_FILE, index=True)

    print(f"Test set saved to: {TEST_SET_FILE}")

    return test_df


def calculate_metrics(model, X_test, y_test):
    """Calculate official held-out test metrics."""

    print()
    print("Generating independent held-out predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }

    return metrics, predictions, probabilities


def save_metrics(metrics, y_test, predictions):
    """Save metrics and classification report."""

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Disease",
            "Disease",
        ],
        output_dict=True,
    )

    output = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    print(f"Metrics saved to: {METRICS_FILE}")


def create_evaluation_plots(
    y_test,
    predictions,
    probabilities,
):
    """Create confusion matrix and ROC curve plots."""

    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        "Final Logistic Regression - Test Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["No Disease", "Disease"],
    )

    plt.yticks(
        [0, 1],
        ["No Disease", "Disease"],
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    confusion_file = (
        PLOTS_DIR
        / "stage3_final_confusion_matrix.png"
    )

    plt.savefig(
        confusion_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "Final Logistic Regression - ROC Curve"
    )

    plt.legend(loc="lower right")

    plt.tight_layout()

    roc_file = (
        PLOTS_DIR
        / "stage3_final_roc_curve.png"
    )

    plt.savefig(
        roc_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Confusion matrix saved to: {confusion_file}")
    print(f"ROC curve saved to: {roc_file}")

    return confusion_file, roc_file


def capture_environment():
    """Capture Python and installed package information."""

    try:
        pip_output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "freeze",
            ],
            text=True,
        )
    except Exception as exc:
        pip_output = (
            f"Unable to execute pip freeze: {exc}"
        )

    environment_text = (
        "MLOps Assignment - Stage 3 Environment\n"
        "========================================\n\n"
        f"Python version:\n{sys.version}\n\n"
        f"Platform:\n{platform.platform()}\n\n"
        f"Python executable:\n{sys.executable}\n\n"
        f"scikit-learn version:\n{sklearn.__version__}\n\n"
        f"MLflow version:\n{mlflow.__version__}\n\n"
        "Installed packages:\n"
        "-------------------\n"
        f"{pip_output}\n"
    )

    with open(
        ENVIRONMENT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(environment_text)

    print(
        f"Environment information saved to: "
        f"{ENVIRONMENT_FILE}"
    )


def create_model_manifest(metrics):
    """
    Create a reproducibility manifest describing
    the model and environment.
    """

    manifest = {
        "project": "heart-disease-mlops",

        "stage": 3,

        "model": {
            "name": "Logistic Regression",
            "file": str(
                MODEL_FILE.relative_to(PROJECT_ROOT)
            ),
            "format": "joblib",
            "pipeline_includes_preprocessing": True,
        },

        "dataset": {
            "source": "UCI Heart Disease Dataset",
            "cleaned_file": str(
                DATA_FILE.relative_to(PROJECT_ROOT)
            ),
            "rows": 303,
            "features": 13,
            "target": TARGET_COLUMN,
            "target_definition": "num > 0 -> 1, otherwise 0",
        },

        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "training_samples": 242,
            "test_samples": 61,
        },

        "model_selection": {
            "criterion": "highest mean 5-fold CV ROC-AUC",
            "selected_model": "Logistic Regression",
            "logistic_regression_cv_roc_auc": 0.9031,
            "random_forest_cv_roc_auc": 0.9020,
        },

        "test_metrics": metrics,

        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        },

        "reproducibility": {
            "random_state": RANDOM_STATE,
            "test_set_file": str(
                TEST_SET_FILE.relative_to(PROJECT_ROOT)
            ),
            "environment_file": str(
                ENVIRONMENT_FILE.relative_to(PROJECT_ROOT)
            ),
        },
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    print(
        f"Model manifest saved to: {MANIFEST_FILE}"
    )

    return manifest


def setup_mlflow():
    """Configure local MLflow tracking."""

    tracking_uri = MLRUNS_DIR.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = (
        "Heart-Disease-Classification"
    )

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(
        experiment_name
    )

    print()
    print("MLflow configuration:")
    print(f"Tracking URI : {tracking_uri}")
    print(f"Experiment   : {experiment_name}")
    print(f"Experiment ID: {experiment_id}")

    return experiment_name


def create_mlflow_final_run(
    model,
    metrics,
    manifest,
    X_test,
    probabilities,
    confusion_file,
    roc_file,
):
    """Create the final consolidated MLflow run."""

    experiment_name = setup_mlflow()

    print()
    print("Creating MLflow final-model run...")

    with mlflow.start_run(
        run_name="Final-Model-Logistic-Regression"
    ) as run:

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "selected_model",
            "Logistic Regression",
        )

        mlflow.log_param(
            "selection_criterion",
            "highest_5_fold_cv_roc_auc",
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "preprocessing",
            "median_imputation + scaling + one_hot_encoding",
        )

        mlflow.log_param(
            "sklearn_version",
            sklearn.__version__,
        )

        mlflow.log_param(
            "mlflow_version",
            mlflow.__version__,
        )

        # ----------------------------------------------------
        # Model hyperparameters
        # ----------------------------------------------------

        mlflow.log_param(
            "logistic_C",
            0.1,
        )

        mlflow.log_param(
            "logistic_solver",
            "liblinear",
        )

        mlflow.log_param(
            "logistic_class_weight",
            "balanced",
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "cv_roc_auc",
            0.9031,
        )

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(
                f"test_{metric_name}",
                metric_value,
            )

        # ----------------------------------------------------
        # Artifacts
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(TEST_SET_FILE),
            artifact_path="reproducibility",
        )

        mlflow.log_artifact(
            str(METRICS_FILE),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(MANIFEST_FILE),
            artifact_path="metadata",
        )

        mlflow.log_artifact(
            str(ENVIRONMENT_FILE),
            artifact_path="environment",
        )

        mlflow.log_artifact(
            str(confusion_file),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(roc_file),
            artifact_path="evaluation",
        )

        if MODEL_METADATA_FILE.exists():
            mlflow.log_artifact(
                str(MODEL_METADATA_FILE),
                artifact_path="metadata",
            )

        # ----------------------------------------------------
        # Model signature
        # ----------------------------------------------------

        # Use floating-point feature columns in the MLflow
        # signature so numeric inputs containing missing values
        # do not conflict with integer-only schema types.
        input_example = X_test.head(3).astype("float64")

        prediction_example = model.predict(input_example)

        signature = infer_signature(
            input_example,
            prediction_example,
        )

        # ----------------------------------------------------
        # Log sklearn pipeline
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "stage",
            "3-final-model",
        )

        mlflow.set_tag(
            "model_status",
            "production_candidate",
        )

        mlflow.set_tag(
            "dataset",
            "UCI Heart Disease",
        )

        mlflow.set_tag(
            "selection",
            "highest_cv_roc_auc",
        )

        run_id = run.info.run_id

    print()
    print("MLflow final run created successfully.")
    print(f"Run ID: {run_id}")
    print(f"Experiment: {experiment_name}")

    return run_id


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 3 - MLFLOW + REPRODUCIBILITY + MODEL PACKAGING")
    print("=" * 70)

    ensure_directories()

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Load final model
    # --------------------------------------------------------

    model = load_final_model()

    # --------------------------------------------------------
    # 3. Recreate deterministic split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = recreate_test_split(df)

    # --------------------------------------------------------
    # 4. Save exact held-out test set
    # --------------------------------------------------------

    save_test_set(
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 5. Independently evaluate final model
    # --------------------------------------------------------

    (
        metrics,
        predictions,
        probabilities,
    ) = calculate_metrics(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 6. Display metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INDEPENDENT HELD-OUT TEST METRICS")
    print("=" * 70)

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name:10s}: "
            f"{metric_value:.4f}"
        )

    # --------------------------------------------------------
    # 7. Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # 8. Generate evaluation plots
    # --------------------------------------------------------

    (
        confusion_file,
        roc_file,
    ) = create_evaluation_plots(
        y_test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------------
    # 9. Capture environment
    # --------------------------------------------------------

    capture_environment()

    # --------------------------------------------------------
    # 10. Create model manifest
    # --------------------------------------------------------

    manifest = create_model_manifest(
        metrics
    )

    # --------------------------------------------------------
    # 11. Create final MLflow run
    # --------------------------------------------------------

    run_id = create_mlflow_final_run(
        model=model,
        metrics=metrics,
        manifest=manifest,
        X_test=X_test,
        probabilities=probabilities,
        confusion_file=confusion_file,
        roc_file=roc_file,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 3 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Final model:")
    print("  Logistic Regression")

    print()
    print("Official held-out test metrics:")

    for metric_name, metric_value in metrics.items():
        print(
            f"  {metric_name:10s}: "
            f"{metric_value:.4f}"
        )

    print()
    print("Created/updated artifacts:")

    print(f"  {TEST_SET_FILE}")
    print(f"  {METRICS_FILE}")
    print(f"  {ENVIRONMENT_FILE}")
    print(f"  {MANIFEST_FILE}")

    print()
    print(f"MLflow Run ID:")
    print(f"  {run_id}")


if __name__ == "__main__":
    main()