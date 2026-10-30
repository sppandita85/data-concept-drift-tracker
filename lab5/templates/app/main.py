import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.onnx"
FEATURES_PATH = APP_DIR / "features.json"

with FEATURES_PATH.open() as f:
    meta = json.load(f)

FEATURES: List[str] = meta["features"]
TARGET: str = meta["target"]
MODEL_TYPE: str = meta["model_type"]
THRESHOLD: float = float(meta.get("threshold", 0.5))
ONNX_INPUT_NAME: str = meta.get("onnx_input_name", "input")

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


class DeviceReading(BaseModel):
    cpu_temp_avg: float = Field(..., ge=0, le=150, description="Average CPU temperature (C)")
    battery_cycles: int = Field(..., ge=0, le=5000)
    device_age_months: int = Field(..., ge=0, le=240)
    disk_errors: int = Field(..., ge=0, le=1000)
    crash_count_30d: int = Field(..., ge=0, le=100)
    battery_health_pct: float = Field(..., ge=0, le=100)
    device_id: Optional[str] = None


def _to_feature_matrix(readings: List[DeviceReading]) -> np.ndarray:
    # Order must match training / ONNX — same as FEATURES in features.json.
    rows = []
    for r in readings:
        row = [
            r.cpu_temp_avg,
            r.battery_cycles,
            r.device_age_months,
            r.disk_errors,
            r.crash_count_30d,
            r.battery_health_pct,
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _run_inference(matrix: np.ndarray) -> np.ndarray:
    try:
        outputs = session.run(None, {input_name: matrix})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    # zipmap=False: probabilities are a plain array (N, 2)
    if len(outputs) >= 2 and hasattr(outputs[1], "shape") and len(outputs[1].shape) == 2:
        probs = outputs[1][:, 1]
    else:
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[:, 1]
    return probs


app = FastAPI(title="Laptop Failure Risk API", version="1.0")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(reading: DeviceReading):
    t0 = time.perf_counter()
    matrix = _to_feature_matrix([reading])
    risk = float(_run_inference(matrix)[0])
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "device_id": reading.device_id,
        "failure_risk": round(risk, 4),
        "high_risk": risk >= THRESHOLD,
        "model_type": MODEL_TYPE,
        "latency_ms": round(latency_ms, 3),
    }


@app.post("/predict/batch")
def predict_batch(readings: List[DeviceReading]):
    if len(readings) == 0:
        raise HTTPException(status_code=422, detail="Batch must contain at least one reading")
    if len(readings) > 1000:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 1000")
    t0 = time.perf_counter()
    matrix = _to_feature_matrix(readings)
    risks = _run_inference(matrix)
    latency_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = latency_ms / len(readings)
    results = []
    for reading, risk in zip(readings, risks):
        risk = float(risk)
        results.append(
            {
                "device_id": reading.device_id,
                "failure_risk": round(risk, 4),
                "high_risk": risk >= THRESHOLD,
                "model_type": MODEL_TYPE,
                "latency_ms": round(per_item_ms, 3),
            }
        )
    return results
