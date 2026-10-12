"""Tests for the saved model and prediction pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.model_utils import ALL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "final_model.joblib"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

TEST_SET_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "test_set.csv"
)


@pytest.fixture(scope="module")
def model():
    """Load the final serialized model."""

    if not MODEL_FILE.is_file():
        pytest.fail(
            f"Model not found: {MODEL_FILE}. "
            "Run Stage 2 training first."
        )

    return joblib.load(MODEL_FILE)


@pytest.fixture(scope="module")
def test_data():
    """Load the saved Stage 3 held-out test set."""

    if not TEST_SET_FILE.is_file():
        pytest.fail(
            f"Test set not found: {TEST_SET_FILE}. "
            "Run Stage 3 first."
        )

    df = pd.read_csv(TEST_SET_FILE)

    assert "target" in df.columns

    X = df[ALL_FEATURES]
    y = df["target"].astype(int)

    return X, y


def test_model_is_sklearn_pipeline(model):
    """The model must include preprocessing and classification."""

    assert isinstance(model, Pipeline)


def test_model_contains_preprocessor_and_classifier(model):
    """Both pipeline stages must be present."""

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_final_classifier_is_logistic_regression(model):
    """Stage 2 selected Logistic Regression."""

    from sklearn.linear_model import LogisticRegression

    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )


def test_model_predicts_binary_classes(model, test_data):
    """Predictions must be binary and match the test-set length."""

    X, y = test_data

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_probability_shape_and_range(model, test_data):
    """Positive-class probabilities must be valid probabilities."""

    X, _ = test_data

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
        atol=1e-6,
    )


def test_model_handles_missing_feature_values(model, test_data):
    """The saved pipeline must impute missing feature values."""

    X, _ = test_data

    sample = X.head(3).copy()

    sample.loc[sample.index[0], "ca"] = np.nan
    sample.loc[sample.index[1], "thal"] = np.nan

    predictions = model.predict(sample)

    assert len(predictions) == len(sample)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_model_handles_unseen_category(model, test_data):
    """Unseen categories should not break prediction."""

    X, _ = test_data

    sample = X.head(1).copy()

    # The fitted encoder should ignore an unseen category.
    sample["cp"] = 999

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in (0, 1)


def test_model_produces_repeatable_predictions(model, test_data):
    """Repeated predictions on identical input should agree."""

    X, _ = test_data

    sample = X.head(10)

    predictions_first = model.predict(sample)
    predictions_second = model.predict(sample)

    np.testing.assert_array_equal(
        predictions_first,
        predictions_second,
    )


def test_model_performance_is_not_degenerate(model, test_data):
    """
    Basic smoke test: the model must make both classes of
    predictions on the held-out dataset.

    This is not a replacement for full model evaluation.
    """

    X, _ = test_data

    predictions = model.predict(X)

    assert len(np.unique(predictions)) == 2