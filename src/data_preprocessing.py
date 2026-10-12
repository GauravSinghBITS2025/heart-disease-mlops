"""
Stage 1B: Data Cleaning

Cleans the locally stored UCI Heart Disease dataset.

Important:
    Model-specific imputation, encoding and scaling are intentionally
    NOT performed here. Those transformations will be implemented
    inside an sklearn Pipeline in Stage 2 to avoid data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = (
    PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

CLEAN_DATA_FILE = (
    PROCESSED_DIR / "heart_disease_clean.csv"
)


# ---------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------

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

TARGET_COLUMN = "num"

BINARY_TARGET_COLUMN = "target"


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_FILE}\n"
            "Run data_acquisition.py first."
        )

    return pd.read_csv(RAW_DATA_FILE)


def replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common textual missing-value markers into NaN.
    """

    missing_markers = ["?", "NA", "N/A", "null", "None", ""]

    df = df.replace(missing_markers, np.nan)

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataset feature columns to numeric values.

    Invalid values are converted to NaN so that they can be
    handled later by the ML preprocessing pipeline.
    """

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the original UCI target:

        0 -> no disease
        1-4 -> disease

    into:

        0 -> no disease
        1 -> disease
    """

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values. "
            "Target values should be investigated before training."
        )

    df[BINARY_TARGET_COLUMN] = (
        df[TARGET_COLUMN] > 0
    ).astype(int)

    return df


def validate_target(df: pd.DataFrame) -> None:
    """Validate the binary target."""

    unique_values = sorted(
        df[BINARY_TARGET_COLUMN].unique()
    )

    if unique_values != [0, 1]:
        raise ValueError(
            f"Unexpected target values: {unique_values}"
        )

    print("\nBinary target distribution:")
    print(
        df[BINARY_TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )


def display_missing_values(df: pd.DataFrame) -> None:
    """Display missing values."""

    missing = df.isna().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    print("\nMissing-value summary:")

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing)


def save_clean_data(df: pd.DataFrame) -> None:
    """Save cleaned dataset."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Drop the original multiclass target.
    output_df = df.drop(
        columns=[TARGET_COLUMN]
    )

    output_df.to_csv(
        CLEAN_DATA_FILE,
        index=False
    )

    print(
        f"\nClean dataset saved to:\n"
        f"{CLEAN_DATA_FILE}"
    )


def main() -> None:

    print("Loading raw dataset...")

    df = load_raw_data()

    print(f"Original shape: {df.shape}")

    print("\nReplacing missing-value markers...")

    df = replace_missing_markers(df)

    print("Converting columns to numeric...")

    df = convert_numeric_columns(df)

    display_missing_values(df)

    print("\nCreating binary target...")

    df = create_binary_target(df)

    validate_target(df)

    save_clean_data(df)

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()