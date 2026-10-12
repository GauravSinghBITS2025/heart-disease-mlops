
"""FastAPI application for serving the packaged heart-disease model."""

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src.model_utils import ALL_FEATURES


# ---------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the model training and packaging steps first."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------

API_REQUESTS = Counter(
    "heart_disease_api_requests_total",
    "Total number of API requests, excluding metrics scrapes",
    ["method", "endpoint", "status"],
)

API_REQUEST_DURATION = Histogram(
    "heart_disease_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps assignment API serving a trained scikit-learn pipeline. "
        "Predictions are for educational use, not medical diagnosis."
    ),
    version="1.1.0",
)


# ---------------------------------------------------------------------
# Request logging and monitoring middleware
# ---------------------------------------------------------------------

@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    """Log requests and record request counts and response latency."""

    method = request.method
    endpoint = request.url.path

    # Avoid recording Prometheus scrapes as application traffic.
    if endpoint == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        logger.exception(
            "API request failed method=%s endpoint=%s",
            method,
            endpoint,
        )
        raise

    finally:
        duration = time.perf_counter() - start_time

        API_REQUESTS.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        API_REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        logger.info(
            "API request method=%s endpoint=%s status=%s "
            "duration_seconds=%.4f",
            method,
            endpoint,
            status_code,
            duration,
        )


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose application metrics in Prometheus text format."""

    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


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
                model.classes_,
                probabilities,
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
        logger.warning(
            "Prediction input could not be processed: %s",
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail=f"Could not process input: {exc}",
        ) from exc
