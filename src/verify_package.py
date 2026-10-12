
"""Verify that the packaged model can be loaded and used independently."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Resolve the repository root from this script's location.
ROOT = Path(__file__).resolve().parents[1]

# Ensure imports work when running: python src/verify_package.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_utils import ALL_FEATURES


MODEL_PATH = ROOT / "artifacts" / "models" / "final_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"


def main():
    for path in (MODEL_PATH, METADATA_PATH, DATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required package file not found: {path}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    missing_features = set(ALL_FEATURES) - set(data.columns)
    if missing_features:
        raise ValueError(
            f"Dataset is missing features: {sorted(missing_features)}"
        )

    # Use a few real rows as a smoke test; preprocessing stays inside the pipeline.
    X = data.loc[:, ALL_FEATURES].head(5).copy()

    if X.empty:
        raise ValueError("No input rows available for the package smoke test.")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    if len(predictions) != len(X):
        raise AssertionError("Prediction count does not match input row count.")

    if probabilities.shape != (len(X), 2):
        raise AssertionError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    if not set(predictions).issubset({0, 1}):
        raise AssertionError("Predictions must be binary values 0 or 1.")

    if not ((probabilities >= 0).all() and (probabilities <= 1).all()):
        raise AssertionError("Probabilities must be between 0 and 1.")

    if not all(
        abs(float(row.sum()) - 1.0) < 1e-6 for row in probabilities
    ):
        raise AssertionError("Each probability row must sum to 1.")

    print("Model package verification: PASSED")
    print(f"Model file: {MODEL_PATH}")
    print(f"Metadata model name: {metadata.get('model_name', 'not specified')}")
    print(f"Input features: {len(ALL_FEATURES)}")
    print(f"Test rows: {len(X)}")
    print(f"Predictions: {predictions.tolist()}")


if __name__ == "__main__":
    main()
