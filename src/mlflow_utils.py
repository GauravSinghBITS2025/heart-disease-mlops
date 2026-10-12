
"""Reusable MLflow logging utilities for the heart disease project."""

import json
import math
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature


EXPERIMENT_NAME = "Heart-Disease-Classification"


def _mlflow_value(value):
    """Convert parameter values into MLflow-compatible scalars or strings."""
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def log_evaluated_model(
    run_name,
    model_name,
    best_params,
    metrics,
    fitted_pipeline,
    input_example,
    artifact_paths=None,
):
    """
    Log one evaluated model to MLflow.

    artifact_paths can contain paths to files or directories, such as
    CV results, evaluation reports, confusion matrices, and ROC plots.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Use the same input columns and order expected by the fitted pipeline.
    example = input_example.head(3).copy().astype("float64")
    predictions = fitted_pipeline.predict(example)
    signature = infer_signature(example, predictions)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_name", model_name)

        mlflow.log_params({
            str(key): _mlflow_value(value)
            for key, value in best_params.items()
        })

        clean_metrics = {}
        for key, value in metrics.items():
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                clean_metrics[str(key)] = numeric_value

        if clean_metrics:
            mlflow.log_metrics(clean_metrics)

        # Log plots, CSVs, JSON reports, and other run-specific evidence.
        for artifact_path in artifact_paths or []:
            path = Path(artifact_path)
            if not path.exists():
                print(f"Skipping missing artifact: {path}")
                continue

            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=path.name)
            else:
                mlflow.log_artifact(str(path), artifact_path=path.parent.name)

        # Each evaluated model run gets its own reusable MLflow model artifact.
        mlflow.sklearn.log_model(
            sk_model=fitted_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=example,
        )

        print(f"Logged MLflow run: {run.info.run_id}")
        return run.info.run_id
