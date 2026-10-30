import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.onnx"
FEATURES_PATH = APP_DIR / "features.json"
LOG_DIR = Path(os.environ.get("LOG_DIR", APP_DIR / "logs"))
PREDICTIONS_LOG = LOG_DIR / "predictions.jsonl"

with FEATURES_PATH.open() as f:
    meta = json.load(f)

FEATURES: List[str] = meta["features"]
TARGET: str = meta["target"]
MODEL_TYPE: str = meta["model_type"]
MODEL_VERSION: str = meta.get("version", meta.get("model_type", "lab5"))
THRESHOLD: float = float(meta.get("threshold", 0.5))
ONNX_INPUT_NAME: str = meta.get("onnx_input_name", "input")

LOG_DIR.mkdir(parents=True, exist_ok=True)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

REQUESTS_TOTAL = Counter(
    "failure_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
ERRORS_TOTAL = Counter("failure_api_errors_total", "Total API errors", ["endpoint"])
LATENCY_SECONDS = Histogram(
    "failure_api_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS_TOTAL = Counter("failure_predictions_total", "Successful /predict calls")
HIGH_RISK_TOTAL = Counter("failure_predictions_high_risk_total", "Predictions flagged high risk")


class DeviceReading(BaseModel):
    cpu_temp_avg: float = Field(..., ge=0, le=150)
    battery_cycles: int = Field(..., ge=0, le=5000)
    device_age_months: int = Field(..., ge=0, le=240)
    disk_errors: int = Field(..., ge=0, le=1000)
    crash_count_30d: int = Field(..., ge=0, le=100)
    battery_health_pct: float = Field(..., ge=0, le=100)
    device_id: Optional[str] = None


def _to_feature_matrix(readings: List[DeviceReading]) -> np.ndarray:
    rows = []
    for r in readings:
        rows.append(
            [
                r.cpu_temp_avg,
                r.battery_cycles,
                r.device_age_months,
                r.disk_errors,
                r.crash_count_30d,
                r.battery_health_pct,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _run_inference(matrix: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {input_name: matrix})
    if len(outputs) >= 2 and hasattr(outputs[1], "shape") and len(outputs[1].shape) == 2:
        return outputs[1][:, 1]
    probs = outputs[0]
    if probs.ndim == 2:
        return probs[:, 1]
    return probs


def _append_prediction_log(reading: DeviceReading, failure_risk: float, high_risk: bool) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": reading.device_id,
        "failure_risk": round(failure_risk, 4),
        "high_risk": high_risk,
        "model_version": MODEL_VERSION,
    }
    for feat in FEATURES:
        record[feat] = getattr(reading, feat)
    with PREDICTIONS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        t0 = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            if response.status_code >= 500:
                ERRORS_TOTAL.labels(endpoint=request.url.path).inc()
            return response
        except Exception:
            ERRORS_TOTAL.labels(endpoint=request.url.path).inc()
            raise
        finally:
            elapsed = time.perf_counter() - t0
            endpoint = request.url.path
            REQUESTS_TOTAL.labels(method=request.method, endpoint=endpoint, status=status).inc()
            LATENCY_SECONDS.labels(endpoint=endpoint).observe(elapsed)


app = FastAPI(title="Laptop Failure Risk API (Lab 7)", version=MODEL_VERSION)
app.add_middleware(MetricsMiddleware)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/view", response_class=HTMLResponse)
def metrics_view():
    """Browser-friendly view (Prometheus still scrapes /metrics as plain text)."""
    body = generate_latest().decode("utf-8")
    escaped = body.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Prometheus metrics</title>
<style>
  body {{ background:#1e1e1e; color:#d4d4d4; font:13px/1.4 ui-monospace, monospace; margin:0; }}
  pre {{ padding:16px; white-space:pre-wrap; word-break:break-all; }}
  a {{ color:#569cd6; }}
</style></head><body>
<p style="padding:16px 16px 0;margin:0">
  Human-readable metrics. Prometheus uses <a href="/metrics">/metrics</a> (plain text).
  Grafana: <a href="http://localhost:3000">localhost:3000</a>
</p>
<pre>{escaped}</pre></body></html>"""


@app.post("/predict")
def predict(reading: DeviceReading):
    t0 = time.perf_counter()
    try:
        matrix = _to_feature_matrix([reading])
        risk = float(_run_inference(matrix)[0])
    except Exception as exc:
        ERRORS_TOTAL.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    high_risk = risk >= THRESHOLD
    _append_prediction_log(reading, risk, high_risk)
    PREDICTIONS_TOTAL.inc()
    if high_risk:
        HIGH_RISK_TOTAL.inc()
    return {
        "device_id": reading.device_id,
        "failure_risk": round(risk, 4),
        "high_risk": high_risk,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency_ms, 3),
    }


@app.post("/predict/batch")
def predict_batch(readings: List[DeviceReading]):
    if not readings:
        raise HTTPException(status_code=422, detail="Batch must contain at least one reading")
    if len(readings) > 1000:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 1000")
    results = []
    for reading in readings:
        results.append(predict(reading))
    return results
