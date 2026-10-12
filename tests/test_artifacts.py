"""Tests for Stage 2 and Stage 3 artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.model_utils import ALL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

MODEL_MANIFEST_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "model_manifest.json"
)

TEST_SET_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "test_set.csv"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "stage3_test_metrics.json"
)

ENVIRONMENT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "environment.txt"
)

CONFUSION_MATRIX_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "plots"
    / "stage3_final_confusion_matrix.png"
)

ROC_CURVE_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "plots"
    / "stage3_final_roc_curve.png"
)


@pytest.mark.parametrize(
    "artifact_path",
    [
        MODEL_FILE,
        MODEL_METADATA_FILE,
        MODEL_MANIFEST_FILE,
        TEST_SET_FILE,
        METRICS_FILE,
        ENVIRONMENT_FILE,
        CONFUSION_MATRIX_FILE,
        ROC_CURVE_FILE,
    ],
)
def test_required_artifact_exists(artifact_path):
    """All expected Stage 2/3 artifacts must exist."""

    assert artifact_path.is_file(), (
        f"Required artifact missing: {artifact_path}"
    )

    assert artifact_path.stat().st_size > 0


def test_test_set_has_expected_dimensions():
    """The saved test set should contain 61 rows and 14 data columns."""

    test_df = pd.read_csv(TEST_SET_FILE)

    assert len(test_df) == 61
    assert set(ALL_FEATURES).issubset(test_df.columns)
    assert "target" in test_df.columns


def test_test_set_target_is_binary():
    """Saved test labels must be valid binary targets."""

    test_df = pd.read_csv(TEST_SET_FILE)

    assert test_df["target"].notna().all()
    assert set(test_df["target"].unique()).issubset({0, 1})


def test_metrics_file_contains_required_metrics():
    """The Stage 3 metrics JSON must be valid and complete."""

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert "metrics" in data

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected_metrics.issubset(
        data["metrics"].keys()
    )

    for metric_name in expected_metrics:
        value = data["metrics"][metric_name]

        assert isinstance(value, (int, float))
        assert 0.0 <= value <= 1.0


def test_manifest_records_reproducibility_settings():
    """The manifest must record the fixed split configuration."""

    with open(
        MODEL_MANIFEST_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert manifest["model"]["name"] == "Logistic Regression"

    assert manifest["split"]["test_size"] == 0.20
    assert manifest["split"]["random_state"] == 42
    assert manifest["split"]["stratified"] is True

    assert (
        manifest["model"]["pipeline_includes_preprocessing"]
        is True
    )


def test_manifest_metrics_match_saved_metrics():
    """Manifest and metrics JSON must describe the same evaluation."""

    with open(
        MODEL_MANIFEST_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    for metric_name, value in report["metrics"].items():
        assert manifest["test_metrics"][metric_name] == pytest.approx(
            value
        )


def test_model_can_be_reloaded():
    """The serialized model must remain loadable."""

    model = joblib.load(MODEL_FILE)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_environment_file_records_versions():
    """Environment capture must include important package versions."""

    content = ENVIRONMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "Python version:" in content
    assert "scikit-learn version:" in content
    assert "MLflow version:" in content