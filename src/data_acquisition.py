"""
Stage 1A: Data Acquisition

Downloads the UCI Heart Disease dataset using the official
ucimlrepo interface and stores a reproducible local CSV copy.

Dataset:
    UCI Heart Disease Dataset
    Dataset ID: 45

Source:
    https://archive.ics.uci.edu/dataset/45/heart+disease
"""

from pathlib import Path
import json

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_FILE = RAW_DIR / "heart_disease.csv"
METADATA_FILE = RAW_DIR / "metadata.json"


# ---------------------------------------------------------------------
# Dataset information
# ---------------------------------------------------------------------

UCI_DATASET_ID = 45

EXPECTED_FEATURES = [
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


def download_dataset() -> pd.DataFrame:
    """
    Download the UCI Heart Disease dataset.

    Returns
    -------
    pd.DataFrame
        Combined feature + target dataframe.
    """

    print("Downloading UCI Heart Disease dataset...")

    heart_disease = fetch_ucirepo(id=UCI_DATASET_ID)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    # Ensure the target has a known name.
    if TARGET_COLUMN not in y.columns:
        y.columns = [TARGET_COLUMN]

    df = pd.concat([X, y], axis=1)

    return df, heart_disease


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Perform basic structural validation.
    """

    print("\nValidating dataset...")

    expected_columns = EXPECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_columns}"
        )

    if len(df) != 303:
        print(
            f"WARNING: Expected approximately 303 rows, "
            f"but received {len(df)}."
        )

    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())


def save_dataset(df: pd.DataFrame, heart_disease) -> None:
    """
    Save the raw dataset and selected metadata.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_DATA_FILE, index=False)

    metadata = {
        "dataset_id": UCI_DATASET_ID,
        "dataset_name": "Heart Disease",
        "source": (
            "UCI Machine Learning Repository - "
            "https://archive.ics.uci.edu/dataset/45/heart+disease"
        ),
        "license": "CC BY 4.0",
        "rows": len(df),
        "columns": list(df.columns),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nDataset saved to:")
    print(RAW_DATA_FILE)

    print(f"\nMetadata saved to:")
    print(METADATA_FILE)


def main() -> None:
    df, heart_disease = download_dataset()

    validate_dataset(df)

    save_dataset(df, heart_disease)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())


if __name__ == "__main__":
    main()