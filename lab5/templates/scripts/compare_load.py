#!/usr/bin/env python3
"""Compare latency: one API container vs load spread across N replicas."""
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

DEFAULT_BASES = [
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
    "http://127.0.0.1:8004",
    "http://127.0.0.1:8005",
    "http://127.0.0.1:8006",
]


def load_rows(csv_path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    sample = df[FEATURES].sample(n=min(n, len(df)), replace=True, random_state=seed)
    return sample.to_dict("records")


def one_request(base: str, row: dict) -> tuple[float | None, str | None]:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base.rstrip('/')}/predict", json=row, timeout=30)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return (time.perf_counter() - t0) * 1000, None
    except requests.RequestException as exc:
        return None, str(exc)


def run_sequential(bases: list[str], rows: list[dict]) -> tuple[list[float], float, int]:
    latencies: list[float] = []
    errors = 0
    t_wall = time.perf_counter()
    for i, row in enumerate(rows):
        base = bases[i % len(bases)]
        ms, err = one_request(base, row)
        if err:
            errors += 1
        else:
            latencies.append(ms)
    return latencies, time.perf_counter() - t_wall, errors


def run_concurrent(
    bases: list[str], rows: list[dict], workers: int
) -> tuple[list[float], float, int]:
    latencies: list[float] = []
    errors = 0
    t_wall = time.perf_counter()

    def task(i_row: tuple[int, dict]) -> tuple[float | None, str | None]:
        i, row = i_row
        base = bases[i % len(bases)]
        return one_request(base, row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, (i, row)) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            ms, err = fut.result()
            if err:
                errors += 1
            else:
                latencies.append(ms)
    return latencies, time.perf_counter() - t_wall, errors


def summarize(name: str, latencies: list[float], wall_s: float, errors: int) -> dict:
    if latencies:
        arr = np.array(latencies)
        stats = {
            "avg_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
    else:
        stats = {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "scenario": name,
        "requests": len(latencies) + errors,
        "ok": len(latencies),
        "errors": errors,
        "total_s": wall_s,
        "rps": len(latencies) / wall_s if wall_s > 0 and latencies else 0,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 1 vs N container load")
    parser.add_argument("--csv", type=Path, default=Path("input/device_telemetry.csv"))
    parser.add_argument("-n", type=int, default=100, help="Requests per scenario")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent clients (parallel mode)")
    parser.add_argument("--mode", choices=("sequential", "parallel", "both"), default="both")
    parser.add_argument("--single-url", default=DEFAULT_BASES[0])
    parser.add_argument("--replica-urls", nargs="+", default=DEFAULT_BASES)
    args = parser.parse_args()

    rows = load_rows(args.csv, args.n, seed=42)
    single = [args.single_url.rstrip("/")]
    replicas = [u.rstrip("/") for u in args.replica_urls]

    print(f"Requests per scenario: {args.n}")
    print(f"Single target: {single[0]}")
    print(f"Replicas ({len(replicas)}): {', '.join(replicas)}")
    print("(Replica mode uses client round-robin — not a production load balancer.)\n")

    results: list[dict] = []

    if args.mode in ("sequential", "both"):
        lat1, w1, e1 = run_sequential(single, rows)
        lat5, w5, e5 = run_sequential(replicas, rows)
        results.append(summarize("sequential → 1 container", lat1, w1, e1))
        results.append(summarize("sequential → N containers (round-robin)", lat5, w5, e5))

    if args.mode in ("parallel", "both"):
        lat1, w1, e1 = run_concurrent(single, rows, args.workers)
        lat5, w5, e5 = run_concurrent(replicas, rows, args.workers)
        results.append(summarize(f"parallel ({args.workers} workers) → 1 container", lat1, w1, e1))
        results.append(
            summarize(
                f"parallel ({args.workers} workers) → N containers (round-robin)", lat5, w5, e5
            )
        )

    df = pd.DataFrame(results)
    cols = [
        "scenario",
        "requests",
        "ok",
        "errors",
        "total_s",
        "rps",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    seq = [r for r in results if r["scenario"].startswith("sequential")]
    if len(seq) == 2 and seq[0]["total_s"] and seq[1]["total_s"] and seq[1]["ok"]:
        speedup = seq[0]["total_s"] / seq[1]["total_s"]
        print(f"\nSequential wall-clock speedup (N replicas vs 1): {speedup:.2f}x")


if __name__ == "__main__":
    main()
