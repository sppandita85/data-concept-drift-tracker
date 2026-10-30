#!/usr/bin/env python3
"""Send predict traffic through the canary router using real device_id values from CSV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Send traffic to router /predict")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if len(df) == 0:
        print("CSV is empty", file=sys.stderr)
        return 1

    counts = {"v1": 0, "v2": 0, "other": 0, "errors": 0}
    for i in range(args.n):
        row = df.iloc[i % len(df)]
        payload = {}
        for f in FEATURES:
            val = row[f]
            if hasattr(val, "item"):
                val = val.item()
            payload[f] = float(val) if f in ("cpu_temp_avg", "battery_health_pct") else int(val)
        payload["device_id"] = str(row.get("device_id", f"row-{i}"))
        try:
            resp = requests.post(args.url, json=payload, timeout=30)
            if resp.status_code >= 400:
                counts["errors"] += 1
                continue
            data = resp.json()
            ver = data.get("routed_version") or data.get("served_by") or data.get("version") or "other"
            if ver in counts:
                counts[ver] += 1
            else:
                counts["other"] += 1
        except requests.RequestException:
            counts["errors"] += 1

    print(json.dumps({"requests": args.n, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
