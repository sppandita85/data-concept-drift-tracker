#!/usr/bin/env python3
"""Periodic drift check: predictions log vs reference.csv → Evidently HTML + Prometheus gauges."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import Gauge, start_http_server

FEATURES = [
    "cpu_temp_avg",
    "battery_cycles",
    "device_age_months",
    "disk_errors",
    "crash_count_30d",
    "battery_health_pct",
]

LOG_PATH = Path(os.environ.get("PREDICTIONS_LOG", "/logs/predictions.jsonl"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_CSV", "/data/reference.csv"))
RETRAIN_TS_PATH = Path(os.environ.get("RETRAIN_TS_FILE", "/data/retrain_last_success_unixtime.txt"))
REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_HTML", "/reports/drift_report.html"))
INTERVAL_SEC = int(os.environ.get("DRIFT_INTERVAL_SEC", "120"))
METRICS_PORT = int(os.environ.get("DRIFT_METRICS_PORT", "9100"))
WINDOW = int(os.environ.get("DRIFT_WINDOW", "1000"))

DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Evidently-derived drift score per feature (0=stable, 1=strong drift)",
    ["feature"],
)
DATASET_DRIFT = Gauge("dataset_drift_share", "Share of columns with detected drift")
RETRAIN_LAST_SUCCESS = Gauge(
    "capstone_retrain_last_success_unixtime",
    "Unix time of last capstone retrain_trigger success (from retrain_last_success_unixtime.txt)",
)


def _load_last_predictions(n: int) -> pd.DataFrame:
    if not LOG_PATH.is_file():
        return pd.DataFrame(columns=FEATURES)
    lines = LOG_PATH.read_text().strip().splitlines()
    rows = []
    for line in lines[-n:]:
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append({f: rec.get(f) for f in FEATURES})
    return pd.DataFrame(rows)


def _drift_scores(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict[str, float], float]:
    mapping = ColumnMapping(numerical_features=FEATURES)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    payload = report.as_dict()
    scores: dict[str, float] = {f: 0.0 for f in FEATURES}
    drift_share = 0.0

    for block in payload.get("metrics", []):
        result = block.get("result", {})
        if "drift_by_columns" in result:
            for col, info in result["drift_by_columns"].items():
                if col in scores:
                    scores[col] = float(info.get("drift_score") or info.get("drift_score_value") or 0.0)
        if "share_of_drifted_columns" in result:
            drift_share = float(result["share_of_drifted_columns"])

    if all(v == 0.0 for v in scores.values()):
        # Fallback: parse column drift flags from dataset drift table
        for block in payload.get("metrics", []):
            result = block.get("result", {})
            for row in result.get("drift_table", []) or []:
                col = row.get("column_name")
                if col in scores:
                    scores[col] = 1.0 if row.get("drift_detected") else 0.0

    return scores, drift_share


def run_once() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    current = _load_last_predictions(WINDOW)
    if len(current) < 30:
        print(f"Skip drift: only {len(current)} predictions (need ~30+)")
        return

    ref = reference[FEATURES].dropna()
    cur = current[FEATURES].dropna()
    scores, drift_share = _drift_scores(ref, cur)
    DATASET_DRIFT.set(drift_share)
    for feat, val in scores.items():
        DRIFT_SCORE.labels(feature=feat).set(val)
    if RETRAIN_TS_PATH.is_file():
        try:
            RETRAIN_LAST_SUCCESS.set(float(RETRAIN_TS_PATH.read_text().strip()))
        except ValueError:
            pass
    print(f"Drift updated; report → {REPORT_PATH}; cpu_temp={scores.get('cpu_temp_avg', 0):.3f}")


def main() -> None:
    start_http_server(METRICS_PORT)
    print(f"Drift metrics on :{METRICS_PORT}; interval {INTERVAL_SEC}s")
    while True:
        try:
            run_once()
        except Exception as exc:
            print("Drift job error:", exc)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
