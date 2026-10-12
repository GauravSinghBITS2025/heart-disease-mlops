"""
Stage 1C: Exploratory Data Analysis

Generates professional EDA plots for the cleaned
UCI Heart Disease dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plots"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


def load_data() -> pd.DataFrame:
    """Load cleaned dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATA_FILE}\n"
            "Run data_preprocessing.py first."
        )

    return pd.read_csv(DATA_FILE)


def prepare_output_directory() -> None:
    """Create output directory."""

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_current_figure(filename: str) -> None:
    """Save and close the current matplotlib figure."""

    output_file = PLOT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


def plot_class_balance(df: pd.DataFrame) -> None:
    """Plot target class distribution."""

    plt.figure(figsize=(7, 5))

    ax = sns.countplot(
        data=df,
        x="target"
    )

    ax.set_title(
        "Heart Disease Class Distribution"
    )

    ax.set_xlabel(
        "Target (0 = No Disease, 1 = Disease)"
    )

    ax.set_ylabel("Number of Patients")

    save_current_figure(
        "01_class_balance.png"
    )


def plot_missing_values(df: pd.DataFrame) -> None:
    """Plot missing values per feature."""

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    plt.figure(figsize=(8, 5))

    if missing.empty:
        plt.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
            fontsize=14
        )

        plt.axis("off")

    else:

        ax = sns.barplot(
            x=missing.values,
            y=missing.index
        )

        ax.set_title(
            "Missing Values by Feature"
        )

        ax.set_xlabel(
            "Number of Missing Values"
        )

        ax.set_ylabel("Feature")

    save_current_figure(
        "02_missing_values.png"
    )


def plot_numeric_distributions(
    df: pd.DataFrame
) -> None:
    """Generate histograms for numerical features."""

    for feature in NUMERIC_FEATURES:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")

        filename = (
            f"03_distribution_{feature}.png"
        )

        save_current_figure(filename)


def plot_correlation_heatmap(
    df: pd.DataFrame
) -> None:
    """
    Plot correlation heatmap for numeric variables.

    This is an exploratory visualization. Categorical integer
    codes should not automatically be interpreted as continuous
    measurements.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title(
        "Correlation Heatmap"
    )

    save_current_figure(
        "04_correlation_heatmap.png"
    )


def plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    filename: str,
    title: str
) -> None:
    """Plot a numerical feature against target."""

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="target",
        y=feature
    )

    plt.title(title)

    plt.xlabel(
        "Target (0 = No Disease, 1 = Disease)"
    )

    plt.ylabel(feature)

    save_current_figure(filename)


def print_eda_summary(
    df: pd.DataFrame
) -> None:
    """Print useful EDA statistics."""

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nTarget distribution:")
    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget proportions:")
    print(
        df["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print("\nMissing values:")
    print(df.isna().sum())


def main() -> None:

    print("Starting EDA...")

    df = load_data()

    prepare_output_directory()

    print_eda_summary(df)

    print("\nGenerating class-balance plot...")
    plot_class_balance(df)

    print("Generating missing-value plot...")
    plot_missing_values(df)

    print("Generating numeric distributions...")

    plot_numeric_distributions(df)

    print("Generating correlation heatmap...")
    plot_correlation_heatmap(df)

    print("Generating feature/target plots...")

    plot_feature_vs_target(
        df,
        "age",
        "05_age_by_target.png",
        "Age Distribution by Heart Disease Status"
    )

    plot_feature_vs_target(
        df,
        "chol",
        "06_cholesterol_by_target.png",
        "Cholesterol Distribution by Heart Disease Status"
    )

    plot_feature_vs_target(
        df,
        "thalach",
        "07_max_heart_rate_by_target.png",
        "Maximum Heart Rate by Heart Disease Status"
    )

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()