import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ".")).resolve()
MODEL_PATH = MODEL_DIR / "model.onnx"
FEATURES_PATH = MODEL_DIR / "features.json"

EXTRA_DELAY_MS = float(os.environ.get("EXTRA_DELAY_MS", "0"))
FAIL_RATE = float(os.environ.get("FAIL_RATE", "0"))

with FEATURES_PATH.open() as f:
    meta = json.load(f)

FEATURES: List[str] = meta["features"]
TARGET: str = meta["target"]
MODEL_TYPE: str = meta.get("model_type", "unknown")
VERSION: str = meta.get("version", MODEL_TYPE)
THRESHOLD: float = float(meta.get("threshold", 0.5))

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


class DeviceReading(BaseModel):
    cpu_temp_avg: float = Field(..., ge=0, le=150)
    battery_cycles: int = Field(..., ge=0, le=5000)
    device_age_months: int = Field(..., ge=0, le=240)
    disk_errors: int = Field(..., ge=0, le=1000)
    crash_count_30d: int = Field(..., ge=0, le=100)
    battery_health_pct: float = Field(..., ge=0, le=100)
    device_id: Optional[str] = None


def _to_feature_matrix(reading: DeviceReading) -> np.ndarray:
    row = [
        reading.cpu_temp_avg,
        reading.battery_cycles,
        reading.device_age_months,
        reading.disk_errors,
        reading.crash_count_30d,
        reading.battery_health_pct,
    ]
    return np.asarray([row], dtype=np.float32)


def _run_inference(matrix: np.ndarray) -> float:
    outputs = session.run(None, {input_name: matrix})
    if len(outputs) >= 2 and hasattr(outputs[1], "shape") and len(outputs[1].shape) == 2:
        return float(outputs[1][0, 1])
    probs = outputs[0]
    if probs.ndim == 2:
        return float(probs[0, 1])
    return float(probs[0])


app = FastAPI(title="Laptop Failure Model Service", version=VERSION)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "model_type": MODEL_TYPE, "features": FEATURES}


@app.post("/predict")
def predict(reading: DeviceReading):
    if FAIL_RATE > 0 and random.random() < FAIL_RATE:
        raise HTTPException(status_code=500, detail="Simulated model failure")
    if EXTRA_DELAY_MS > 0:
        time.sleep(EXTRA_DELAY_MS / 1000.0)

    t0 = time.perf_counter()
    matrix = _to_feature_matrix(reading)
    try:
        risk = _run_inference(matrix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "device_id": reading.device_id,
        "failure_risk": round(risk, 4),
        "high_risk": risk >= THRESHOLD,
        "model_type": MODEL_TYPE,
        "version": VERSION,
        "served_by": VERSION,
        "latency_ms": round(latency_ms, 3),
    }
