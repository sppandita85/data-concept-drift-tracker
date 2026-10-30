import hashlib
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

CONFIG_PATH = Path(os.environ.get("ROUTER_CONFIG", Path(__file__).resolve().parent / "router_config.json"))
LOG_DIR = Path(os.environ.get("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))
REQUESTS_LOG = LOG_DIR / "router_requests.jsonl"
ROLLBACK_LOG = LOG_DIR / "rollback_events.jsonl"

lock = threading.Lock()
recent_v1: Deque[Dict[str, Any]] = deque(maxlen=200)
recent_v2: Deque[Dict[str, Any]] = deque(maxlen=200)
v2_requests_since_check = 0
rolled_back = False


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return json.load(f)


def _save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _pick_version(device_id: str, v2_percent: int) -> str:
    if v2_percent <= 0:
        return "v1"
    if v2_percent >= 100:
        return "v2"
    bucket = int(hashlib.md5(device_id.encode("utf-8")).hexdigest(), 16) % 100
    return "v2" if bucket < v2_percent else "v1"


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def _summarize(records: Deque[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"count": 0, "error_rate_pct": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "high_risk_pct": 0.0}
    latencies = [float(r["latency_ms"]) for r in records]
    errors = sum(1 for r in records if r.get("status_code", 200) >= 400)
    high_risk = sum(1 for r in records if r.get("high_risk"))
    return {
        "count": len(records),
        "error_rate_pct": round(100.0 * errors / len(records), 2),
        "p50_ms": round(_percentile(latencies, 50), 2),
        "p95_ms": round(_percentile(latencies, 95), 2),
        "high_risk_pct": round(100.0 * high_risk / len(records), 2),
    }


def _maybe_rollback(cfg: Dict[str, Any]) -> None:
    global v2_requests_since_check, rolled_back
    if not cfg.get("auto_rollback", True):
        return
    every_n = int(cfg.get("rollback_check_every_n_v2", 50))
    if v2_requests_since_check < every_n:
        return
    v2_requests_since_check = 0
    summary = _summarize(recent_v2)
    p95_limit = float(cfg.get("p95_limit_ms", 200))
    err_limit = float(cfg.get("error_limit_pct", 1.0))
    if summary["count"] == 0:
        return
    if summary["p95_ms"] > p95_limit or summary["error_rate_pct"] > err_limit:
        cfg["v2_percent"] = 0
        _save_config(cfg)
        rolled_back = True
        _append_jsonl(
            ROLLBACK_LOG,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "p95_or_error_limit",
                "v2_p95_ms": summary["p95_ms"],
                "v2_error_rate_pct": summary["error_rate_pct"],
                "p95_limit_ms": p95_limit,
                "error_limit_pct": err_limit,
            },
        )


class PredictBody(BaseModel):
    cpu_temp_avg: float = Field(..., ge=0, le=150)
    battery_cycles: int = Field(..., ge=0, le=5000)
    device_age_months: int = Field(..., ge=0, le=240)
    disk_errors: int = Field(..., ge=0, le=1000)
    crash_count_30d: int = Field(..., ge=0, le=100)
    battery_health_pct: float = Field(..., ge=0, le=100)
    device_id: Optional[str] = None


class ConfigUpdate(BaseModel):
    v2_percent: int = Field(..., ge=0, le=100)


app = FastAPI(title="Canary Router", version="1.0")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    cfg = _load_config()
    return {"status": "ok", "v2_percent": cfg.get("v2_percent", 0)}


@app.get("/metrics")
def metrics():
    with lock:
        cfg = _load_config()
        return {
            "v2_percent": cfg.get("v2_percent", 0),
            "rolled_back": rolled_back,
            "v1": _summarize(recent_v1),
            "v2": _summarize(recent_v2),
        }


@app.post("/config")
def update_config(body: ConfigUpdate):
    global rolled_back
    with lock:
        cfg = _load_config()
        cfg["v2_percent"] = body.v2_percent
        _save_config(cfg)
        if body.v2_percent > 0:
            rolled_back = False
    return {"v2_percent": body.v2_percent}


@app.post("/predict")
def predict(body: PredictBody):
    global v2_requests_since_check
    device_id = body.device_id or "unknown"
    payload = body.model_dump()

    with lock:
        cfg = _load_config()
        version = _pick_version(device_id, int(cfg.get("v2_percent", 0)))
        base = cfg["v2_url"] if version == "v2" else cfg["v1_url"]

    url = f"{base.rstrip('/')}/predict"
    t0 = time.perf_counter()
    status_code = 200
    response_data: Dict[str, Any] = {}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            status_code = resp.status_code
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            response_data = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        status_code = 502
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_id": device_id,
            "routed_version": version,
            "latency_ms": round(latency_ms, 3),
            "status_code": status_code,
            "high_risk": bool(response_data.get("high_risk")) if response_data else False,
            "failure_risk": response_data.get("failure_risk"),
        }
        with lock:
            if version == "v2":
                recent_v2.append(record)
                v2_requests_since_check += 1
                cfg = _load_config()
                _maybe_rollback(cfg)
            else:
                recent_v1.append(record)
            _append_jsonl(REQUESTS_LOG, record)

    out = dict(response_data)
    out["routed_version"] = version
    out["router_latency_ms"] = round(latency_ms, 3)
    return out
