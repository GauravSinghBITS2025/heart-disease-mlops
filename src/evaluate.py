from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "final_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "model_metadata.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "plots"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
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


def main():

    print("=" * 70)
    print("STAGE 2 - FINAL MODEL EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print("\nLoading final model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["target"]

    print(f"Dataset shape: {df.shape}")

    # -------------------------------------------------------------
    # IMPORTANT
    # -------------------------------------------------------------
    #
    # The model was already evaluated against the held-out test
    # set during train.py.
    #
    # This script provides a reusable evaluation utility, but
    # because the exact test split was created inside train.py,
    # we should not pretend that evaluating on the full dataset
    # is an unbiased test-set evaluation.
    #
    # Therefore, this script reports predictions on the complete
    # dataset as an additional diagnostic only.
    #
    # The official test metrics are stored in model_metadata.json.
    # -------------------------------------------------------------

    print(
        "\nGenerating diagnostic predictions "
        "on the complete cleaned dataset..."
    )

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(
            y,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    print("\nFull-dataset diagnostic metrics:")

    for name, value in metrics.items():
        print(f"{name:10s}: {value:.4f}")

    # -------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------

    report = classification_report(
        y,
        y_pred,
        target_names=[
            "No Disease",
            "Disease",
        ],
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    report_path = REPORT_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_title("Final Model - Diagnostic Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No Disease", "Disease"]
    )

    ax.set_yticklabels(
        ["No Disease", "Disease"]
    )

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

    confusion_path = (
        PLOT_DIR
        / "final_model_confusion_matrix_diagnostic.png"
    )

    fig.savefig(
        confusion_path,
        dpi=150,
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # ROC curve
    # -------------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y,
        y_prob,
    )

    auc = roc_auc_score(
        y,
        y_prob,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"Final Model (AUC = {auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_title(
        "Final Model - Diagnostic ROC Curve"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    roc_path = (
        PLOT_DIR
        / "final_model_roc_curve_diagnostic.png"
    )

    fig.savefig(
        roc_path,
        dpi=150,
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Display official test metrics
    # -------------------------------------------------------------

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        print("\n" + "=" * 70)
        print("OFFICIAL HELD-OUT TEST METRICS")
        print("=" * 70)

        print(
            f"Final model: "
            f"{metadata['model']}"
        )

        print(
            f"CV ROC-AUC: "
            f"{metadata['best_cv_roc_auc']:.4f}"
        )

        for name, value in metadata[
            "test_metrics"
        ].items():
            print(
                f"{name:10s}: "
                f"{value:.4f}"
            )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()