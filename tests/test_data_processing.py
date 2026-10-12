"""Tests for the cleaned Heart Disease dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_utils import (
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    create_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

EXPECTED_ROWS = 303


@pytest.fixture(scope="module")
def dataset():
    """Load the cleaned dataset once for this test module."""

    if not DATA_FILE.exists():
        pytest.fail(
            f"Cleaned dataset not found: {DATA_FILE}. "
            "Run the Stage 1 scripts first."
        )

    return pd.read_csv(DATA_FILE)


def test_cleaned_dataset_exists():
    """The cleaned dataset must exist."""

    assert DATA_FILE.is_file()


def test_dataset_has_expected_number_of_rows(dataset):
    """The UCI Cleveland dataset contains 303 records."""

    assert len(dataset) == EXPECTED_ROWS


def test_dataset_contains_expected_features(dataset):
    """All expected input features and target must be present."""

    assert set(ALL_FEATURES).issubset(dataset.columns)
    assert "target" in dataset.columns


def test_dataset_has_no_duplicate_column_names(dataset):
    """Column names must be unique."""

    assert dataset.columns.is_unique


def test_target_contains_only_binary_values(dataset):
    """The transformed target must contain only 0 and 1."""

    assert dataset["target"].notna().all()
    assert set(dataset["target"].unique()).issubset({0, 1})


def test_target_has_both_classes(dataset):
    """Both classes must be represented."""

    assert dataset["target"].nunique() == 2


def test_dataset_has_expected_missing_values(dataset):
    """
    The current Stage 1 design retains missing ca and thal
    values for imputation inside the model pipeline.
    """

    assert dataset["ca"].isna().sum() == 4
    assert dataset["thal"].isna().sum() == 2


def test_missing_values_are_confined_to_expected_features(dataset):
    """No other features or the target should have missing values."""

    permitted_missing = {"ca", "thal", "target"}

    for column in dataset.columns:
        if column not in permitted_missing:
            assert dataset[column].isna().sum() == 0, (
                f"Unexpected missing values in {column}"
            )

    assert dataset["target"].notna().all()


def test_feature_groups_are_consistent():
    """Feature groups must match the complete model feature list."""

    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    assert set(NUMERICAL_FEATURES).isdisjoint(
        set(CATEGORICAL_FEATURES)
    )

    assert set(NUMERICAL_FEATURES) | set(
        CATEGORICAL_FEATURES
    ) == set(ALL_FEATURES)


def test_preprocessor_can_handle_missing_values(dataset):
    """
    The preprocessing pipeline must handle the missing values
    retained by Stage 1.
    """

    X = dataset[ALL_FEATURES].head(20).copy()

    preprocessor = create_preprocessor(
        scale_numeric=True
    )

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories(dataset):
    """
    OneHotEncoder(handle_unknown='ignore') should tolerate
    an unseen category at inference time.
    """

    X_train = dataset[ALL_FEATURES].iloc[:100].copy()
    X_new = dataset[ALL_FEATURES].iloc[100:101].copy()

    # Introduce a valid but previously unseen category for cp.
    X_train["cp"] = X_train["cp"].astype(int)
    X_new["cp"] = 999

    preprocessor = create_preprocessor(
        scale_numeric=True
    )

    preprocessor.fit(X_train)
    transformed = preprocessor.transform(X_new)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()