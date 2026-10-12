
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

try:
    # Supports: python -m src.train
    from .model_utils import (
        ALL_FEATURES,
        create_logistic_regression_pipeline,
        create_random_forest_pipeline,
    )
except ImportError:
    # Supports: python src/train.py
    from model_utils import (
        ALL_FEATURES,
        create_logistic_regression_pipeline,
        create_random_forest_pipeline,
    )


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"
PLOT_DIR = ARTIFACTS_DIR / "plots"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

for directory in (
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    MLRUNS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "Heart-Disease-Classification"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "roc_auc"


# ---------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------

mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(
    tracking_uri=MLRUNS_DIR.resolve().as_uri()
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_model_name(model_name: str) -> str:
    """Convert a display name to a filesystem-safe name."""
    return model_name.lower().replace(" ", "_")


def json_default(value):
    """Convert common non-JSON-native values into serializable values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """Calculate binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)

    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve(
    y_true,
    y_prob,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_title(f"{model_name} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_on_test_set(
    model,
    X_test,
    y_test,
    model_name: str,
):
    """Evaluate a fitted pipeline and save its evaluation plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    name = safe_model_name(model_name)
    confusion_path = PLOT_DIR / f"{name}_confusion_matrix.png"
    roc_path = PLOT_DIR / f"{name}_roc_curve.png"

    save_confusion_matrix(
        y_test, y_pred, model_name, confusion_path
    )
    save_roc_curve(
        y_test, y_prob, model_name, roc_path
    )

    return metrics, confusion_path, roc_path


def write_json(data: dict, output_path: Path) -> None:
    """Write a JSON document in a consistent format."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            default=json_default,
        )


# ---------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TASK 2 + TASK 3: MODEL DEVELOPMENT AND EXPERIMENT TRACKING")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load and validate data
    # -------------------------------------------------------------

    print("\nLoading cleaned dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run the data acquisition and preprocessing scripts first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    missing_features = [
        feature for feature in ALL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected features: {missing_features}"
        )

    if "target" not in df.columns:
        raise ValueError("Target column 'target' not found.")

    X = df.loc[:, ALL_FEATURES].copy()
    y = df["target"].copy()

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Expected binary target values 0 and 1; got {sorted(y.unique())}"
        )

    print(f"Number of features: {len(ALL_FEATURES)}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # -------------------------------------------------------------
    # Stratified train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")

    # -------------------------------------------------------------
    # Cross-validation and model configurations
    # -------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_configs = {
        "Logistic Regression": {
            "pipeline": create_logistic_regression_pipeline(),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "classifier__solver": ["liblinear", "lbfgs"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "pipeline": create_random_forest_pipeline(),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__max_features": ["sqrt"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    }

    results = []
    best_models = {}

    # -------------------------------------------------------------
    # Train, tune, evaluate and log each model
    # -------------------------------------------------------------

    for model_name, config in model_configs.items():
        print("\n" + "=" * 70)
        print(f"Training: {model_name}")
        print("=" * 70)

        pipeline = config["pipeline"]
        param_grid = config["param_grid"]
        name = safe_model_name(model_name)

        with mlflow.start_run(
            run_name=f"{model_name} - Hyperparameter Tuning"
        ) as run:

            # Log general experiment parameters.
            mlflow.set_tags({
                "assignment_task": "3 - Experiment Tracking",
                "model_family": model_name,
                "dataset": "UCI Heart Disease",
                "selection_metric": SCORING,
            })

            mlflow.log_params({
                "model": model_name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_folds": CV_FOLDS,
                "scoring": SCORING,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(ALL_FEATURES),
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
            })

            # Store the search space as an artifact so it is preserved
            # even if its JSON representation is too long for a parameter.
            search_space_path = REPORT_DIR / f"{name}_search_space.json"
            write_json(param_grid, search_space_path)
            mlflow.log_artifact(
                str(search_space_path),
                artifact_path="configuration",
            )

            # Run grid search. Preprocessing is inside each pipeline,
            # so each CV fold fits its own preprocessing transformers.
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=SCORING,
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
                error_score="raise",
            )

            print("\nRunning hyperparameter search...")
            grid_search.fit(X_train, y_train)

            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

            # Log selected hyperparameters.
            mlflow.log_params({
                key: (
                    "None" if value is None else str(value)
                )
                for key, value in grid_search.best_params_.items()
            })

            mlflow.log_metric(
                "best_cv_roc_auc",
                float(grid_search.best_score_),
            )

            # Save and log all CV search results.
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results_path = REPORT_DIR / f"{name}_cv_results.csv"
            cv_results.to_csv(cv_results_path, index=False)

            mlflow.log_artifact(
                str(cv_results_path),
                artifact_path="cv_results",
            )

            # Evaluate on the held-out test set.
            test_metrics, confusion_path, roc_path = (
                evaluate_on_test_set(
                    grid_search.best_estimator_,
                    X_test,
                    y_test,
                    model_name,
                )
            )

            print("\nHeld-out test metrics:")
            for metric_name, metric_value in test_metrics.items():
                print(f"{metric_name:10s}: {metric_value:.4f}")
                mlflow.log_metric(
                    f"test_{metric_name}",
                    float(metric_value),
                )

            # Log evaluation plots to this model's MLflow run.
            mlflow.log_artifact(
                str(confusion_path),
                artifact_path="plots",
            )
            mlflow.log_artifact(
                str(roc_path),
                artifact_path="plots",
            )

            # Save a per-run evaluation report.
            run_report_path = REPORT_DIR / f"{name}_metrics.json"
            run_report = {
                "model": model_name,
                "best_params": grid_search.best_params_,
                "best_cv_roc_auc": float(grid_search.best_score_),
                "test_metrics": test_metrics,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "random_state": RANDOM_STATE,
                "cv_folds": CV_FOLDS,
                "scoring": SCORING,
            }
            write_json(run_report, run_report_path)

            mlflow.log_artifact(
                str(run_report_path),
                artifact_path="evaluation",
            )

            # Log the complete fitted pipeline, including preprocessing.
            # Infer the signature using numeric input columns to avoid
            # the integer/missing-value schema warning.
            input_example = X_test.head(3).copy().astype("float64")
            prediction_example = grid_search.best_estimator_.predict(
                input_example
            )
            signature = infer_signature(
                input_example,
                prediction_example,
            )

            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path="model",
                signature=signature,
            )

            # Keep results for final comparison and packaging.
            results.append({
                "model": model_name,
                "best_cv_roc_auc": float(grid_search.best_score_),
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                },
                "best_params": json.dumps(
                    grid_search.best_params_,
                    default=json_default,
                    sort_keys=True,
                ),
                "run_id": run.info.run_id,
            })

            best_models[model_name] = {
                "model": grid_search.best_estimator_,
                "cv_roc_auc": float(grid_search.best_score_),
                "test_metrics": test_metrics,
                "best_params": grid_search.best_params_,
                "run_id": run.info.run_id,
            }

            print(f"MLflow run ID: {run.info.run_id}")

    # -------------------------------------------------------------
    # Compare models using cross-validation performance
    # -------------------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="best_cv_roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "best_cv_roc_auc",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
            ]
        ].to_string(index=False)
    )

    # Save a visual comparison of the CV ROC-AUC scores.
    comparison_plot_path = PLOT_DIR / "model_comparison_cv_roc_auc.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        results_df["model"],
        results_df["best_cv_roc_auc"],
    )
    ax.set_title("Model Comparison - Cross-Validation ROC-AUC")
    ax.set_ylabel("Mean CV ROC-AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(comparison_plot_path, dpi=150)
    plt.close(fig)

    # Attach the final comparison evidence to both evaluated model runs.
    # This is done after the runs close so both retain the shared reports.
    for model_info in best_models.values():
        run_id = model_info["run_id"]

        client.log_artifact(
            run_id,
            str(results_path),
            artifact_path="model_comparison",
        )
        client.log_artifact(
            run_id,
            str(comparison_plot_path),
            artifact_path="model_comparison",
        )

    # -------------------------------------------------------------
    # Select the final model using CV ROC-AUC only
    # -------------------------------------------------------------

    best_model_name = results_df.iloc[0]["model"]
    selected = best_models[best_model_name]
    final_model = selected["model"]

    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTION")
    print("=" * 70)
    print(f"Selected model: {best_model_name}")
    print("Selection criterion: highest mean CV ROC-AUC")

    # -------------------------------------------------------------
    # Save the reusable fitted pipeline
    # -------------------------------------------------------------

    final_model_path = MODEL_DIR / "final_model.joblib"
    joblib.dump(final_model, final_model_path)

    print(f"\nFinal model saved to: {final_model_path}")

    # -------------------------------------------------------------
    # Save metadata for reproducibility
    # -------------------------------------------------------------

    metadata = {
        "model": best_model_name,
        "model_name": best_model_name,
        "mlflow_experiment": EXPERIMENT_NAME,
        "selected_mlflow_run_id": selected["run_id"],
        "selection_metric": "best_cv_roc_auc",
        "best_cv_roc_auc": selected["cv_roc_auc"],
        "best_params": selected["best_params"],
        "test_metrics": selected["test_metrics"],
        "features": ALL_FEATURES,
        "numerical_features": [
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
        ],
        "categorical_features": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
        "target": "target",
        "target_definition": "num > 0 maps to 1; num == 0 maps to 0",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    write_json(metadata, metadata_path)

    # Log packaging evidence to both model runs.
    # The selected run additionally receives the exact joblib package.
    for model_name, model_info in best_models.items():
        run_id = model_info["run_id"]

        client.log_artifact(
            run_id,
            str(metadata_path),
            artifact_path="reproducibility",
        )

        if model_name == best_model_name:
            client.log_artifact(
                run_id,
                str(final_model_path),
                artifact_path="packaged_model",
            )

    print(f"Model metadata saved to: {metadata_path}")

    # -------------------------------------------------------------
    # Completion summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EXPERIMENT TRACKING COMPLETED")
    print("=" * 70)
    print(f"MLflow tracking directory: {MLRUNS_DIR}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Selected model: {best_model_name}")
    print(f"Model file: {final_model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Comparison report: {results_path}")
    print(f"Comparison plot: {comparison_plot_path}")


if __name__ == "__main__":
    main()
