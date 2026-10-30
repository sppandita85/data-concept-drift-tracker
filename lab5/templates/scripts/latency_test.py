#!/usr/bin/env python3
"""Measure /predict or /predict/batch latency against a running API."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FEATURES = [
    "cpu_temp_avg",
    "battery_cycles",
    "device_age_months",
    "disk_errors",
    "crash_count_30d",
    "battery_health_pct",
]


def percentile(values, p: float) -> float:
    return float(np.percentile(values, p))


def load_rows(csv_path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    sample = df[FEATURES].sample(n=min(n, len(df)), replace=True, random_state=seed)
    return sample.to_dict("records")


def warmup(base: str, rows: list[dict], count: int, endpoint: str) -> None:
    if count <= 0 or not rows:
        return
    for i in range(count):
        row = rows[i % len(rows)]
        _send(base, endpoint, row, batch_size=1, quiet=True)


def _send(
    base: str,
    endpoint: str,
    row: dict,
    batch_size: int,
    quiet: bool = False,
) -> tuple[float | None, str | None]:
    """Returns (latency_ms, error_message)."""
    t0 = time.perf_counter()
    try:
        if endpoint == "predict":
            r = requests.post(f"{base}/predict", json=row, timeout=30)
        else:
            payload = [row] * batch_size
            r = requests.post(f"{base}/predict/batch", json=payload, timeout=60)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return (time.perf_counter() - t0) * 1000, None
    except requests.RequestException as exc:
        return None, str(exc)


def run_load(
    base: str,
    rows: list[dict],
    endpoint: str,
    workers: int,
    batch_size: int,
) -> tuple[list[float], int]:
    latencies: list[float] = []
    errors = 0

    if workers <= 1:
        for row in rows:
            ms, err = _send(base, endpoint, row, batch_size)
            if err:
                errors += 1
            else:
                latencies.append(ms)
        return latencies, errors

    def task(row: dict) -> tuple[float | None, str | None]:
        return _send(base, endpoint, row, batch_size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, row) for row in rows]
        for fut in as_completed(futures):
            ms, err = fut.result()
            if err:
                errors += 1
            else:
                latencies.append(ms)
    return latencies, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test /predict or /predict/batch")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with feature columns")
    parser.add_argument("-n", type=int, default=500, help="Number of measured requests")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests (not counted)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel clients (1 = sequential)")
    parser.add_argument(
        "--endpoint",
        choices=("predict", "batch"),
        default="predict",
        help="API route to hit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Rows per /predict/batch call when --endpoint batch",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    base = args.url.rstrip("/")
    rows = load_rows(args.csv, max(args.n, args.warmup), seed=42)
    warmup(base, rows, args.warmup, args.endpoint)
    measure_rows = rows[: args.n]

    t_start = time.perf_counter()
    latencies, errors = run_load(
        base, measure_rows, args.endpoint, args.workers, args.batch_size
    )
    total = time.perf_counter() - t_start

    print(f"URL: {base}")
    print(f"Endpoint: /{args.endpoint}" + (f" (batch_size={args.batch_size})" if args.endpoint == "batch" else ""))
    print(f"Workers: {args.workers} | Warmup: {args.warmup}")
    print(f"Requests attempted: {args.n} | OK: {len(latencies)} | Errors: {errors}")

    if not latencies:
        print("No successful requests — check URL and server logs.")
        raise SystemExit(1)

    arr = np.array(latencies)
    print(f"p50 (ms): {percentile(arr, 50):.2f}")
    print(f"p95 (ms): {percentile(arr, 95):.2f}")
    print(f"p99 (ms): {percentile(arr, 99):.2f}")
    print(f"avg (ms): {arr.mean():.2f}")
    print(f"RPS: {len(latencies) / total:.2f}")


if __name__ == "__main__":
    main()
