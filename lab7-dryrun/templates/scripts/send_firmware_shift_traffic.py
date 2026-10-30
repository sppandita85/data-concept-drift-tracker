#!/usr/bin/env python3
"""Step 9: simulate firmware update — CPU temperature +8°C on every reading."""
from __future__ import annotations

import argparse
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
TEMP_SHIFT_C = 8.0


def _payload(row, shift: float) -> dict:
    out = {}
    for f in FEATURES:
        val = row[f]
        if hasattr(val, "item"):
            val = val.item()
        out[f] = float(val) if f in ("cpu_temp_avg", "battery_health_pct") else int(val)
    out["cpu_temp_avg"] = min(150.0, out["cpu_temp_avg"] + shift)
    out["device_id"] = str(row.get("device_id", "unknown"))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    p.add_argument("--csv", type=Path, default=Path("input/device_telemetry.csv"))
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--shift", type=float, default=TEMP_SHIFT_C)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    ok = 0
    for i in range(args.n):
        row = df.iloc[i % len(df)]
        try:
            r = requests.post(args.url, json=_payload(row, args.shift), timeout=15)
            if r.status_code < 400:
                ok += 1
        except requests.RequestException:
            pass
    print(f"Sent {args.n} shifted requests (+{args.shift}°C cpu_temp_avg); {ok} succeeded")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
