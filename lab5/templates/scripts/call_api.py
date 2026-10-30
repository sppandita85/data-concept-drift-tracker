#!/usr/bin/env python3
"""Simple client for the laptop failure API."""
import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description="Call laptop failure risk API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=10)
    print("GET /health", health.status_code)
    print(json.dumps(health.json(), indent=2))

    payload = {
        "device_id": "PF00007",
        "cpu_temp_avg": 73.7,
        "battery_cycles": 611,
        "device_age_months": 42,
        "disk_errors": 4,
        "crash_count_30d": 2,
        "battery_health_pct": 71.0,
    }
    pred = requests.post(f"{base}/predict", json=payload, timeout=10)
    print("\nPOST /predict", pred.status_code)
    print(json.dumps(pred.json(), indent=2))


if __name__ == "__main__":
    main()
