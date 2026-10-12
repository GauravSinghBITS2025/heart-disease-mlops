
"""FastAPI application for serving the packaged heart-disease model."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.model_utils import ALL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the model training and packaging steps first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps assignment API serving a trained scikit-learn pipeline. "
        "Predictions are for educational use, not medical diagnosis."
    ),
    version="1.0.0",
)


class HeartDiseaseInput(BaseModel):
    """Validate the 13 features used by the trained model."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(gt=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=7)


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: HeartDiseaseInput):
    """Return a binary prediction and the model's class probabilities."""

    try:
        # Keep feature names and order consistent with model training.
        input_df = pd.DataFrame(
            [request.model_dump()],
            columns=ALL_FEATURES,
        )

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        probability_by_class = {
            str(class_label): float(probability)
            for class_label, probability in zip(
                model.classes_, probabilities
            )
        }

        return {
            "prediction": prediction,
            "prediction_label": (
                "disease_detected"
                if prediction == 1
                else "no_disease_detected"
            ),
            "class_probabilities": probability_by_class,
            "confidence": float(max(probabilities)),
            "disclaimer": (
                "Educational MLOps demonstration only; "
                "not a medical diagnosis."
            ),
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process input: {exc}",
        ) from exc
