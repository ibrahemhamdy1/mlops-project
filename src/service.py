"""Inference service — FastAPI + Prometheus metrics.

Endpoints:
  POST /predict   — score one or more samples (30 features each)
  GET  /healthz   — liveness (dumb: process responds)
  GET  /readyz    — readiness (model loaded)
  GET  /metrics   — Prometheus scrape endpoint

The Prometheus metrics include prediction-distribution counters so the
monitoring stack can watch for prediction drift, not just latency.
"""
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

MODEL_PATH = Path("model/model.pkl")
N_FEATURES = 30

PREDICTIONS = Counter(
    "model_predictions_total", "Predictions served", ["predicted_class"]
)
LATENCY = Histogram("model_inference_seconds", "Inference latency")
CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Predicted probability distribution (drift signal)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

app = FastAPI(title="fraud-detector-style inference service")
model = None


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(..., description="rows of 30 features")


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]


@app.on_event("startup")
def load_model() -> None:
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    x = np.asarray(req.instances, dtype=float)
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=422, detail=f"each instance must have {N_FEATURES} features"
        )

    start = time.perf_counter()
    proba = model.predict_proba(x)[:, 1]
    LATENCY.observe(time.perf_counter() - start)

    preds = (proba >= 0.5).astype(int)
    for p, pr in zip(preds, proba):
        PREDICTIONS.labels(predicted_class=str(int(p))).inc()
        CONFIDENCE.observe(float(pr))

    return PredictResponse(
        predictions=preds.tolist(),
        probabilities=[round(float(p), 4) for p in proba],
    )
