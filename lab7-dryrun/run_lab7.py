#!/usr/bin/env python3
"""
Lab 7 — monitoring & drift (local Docker).

  python lab7-dryrun/prepare_lab7.py
  python lab7-dryrun/run_lab7.py prepare
  python lab7-dryrun/run_lab7.py up
  python lab7-dryrun/run_lab7.py smoke | traffic | firmware | drift-once | down
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import requests

LAB7_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB7_DIR.parent
OUT = LAB7_DIR / "outputs"
API_BASE = __import__("os").environ.get("LAB7_API_URL", "http://127.0.0.1:8088")


def _compose(*args: str) -> int:
    return subprocess.call(["docker", "compose", *args], cwd=OUT)


def cmd_prepare() -> int:
    return subprocess.call([sys.executable, str(LAB7_DIR / "prepare_lab7.py")])


def cmd_smoke() -> int:
    url = f"{API_BASE.rstrip('/')}/predict"
    sample = {
        "device_id": "smoke-1",
        "cpu_temp_avg": 55.0,
        "battery_cycles": 400,
        "device_age_months": 24,
        "disk_errors": 0,
        "crash_count_30d": 1,
        "battery_health_pct": 88.0,
    }
    for i in range(3):
        sample["device_id"] = f"smoke-{i}"
        r = requests.post(url, json=sample, timeout=10)
        r.raise_for_status()
    log = OUT / "logs" / "predictions.jsonl"
    lines = log.read_text().strip().splitlines() if log.is_file() else []
    print(f"predictions.jsonl lines: {len(lines)}")
    m = requests.get(f"{API_BASE.rstrip('/')}/metrics", timeout=10)
    m.raise_for_status()
    print("metrics sample:\n", "\n".join(m.text.splitlines()[:12]))
    return 0


def cmd_traffic(firmware: bool) -> int:
    script = OUT / "scripts" / (
        "send_firmware_shift_traffic.py" if firmware else "send_normal_traffic.py"
    )
    csv = REPO_ROOT / "input" / "device_telemetry.csv"
    url = f"{API_BASE.rstrip('/')}/predict"
    return subprocess.call(
        [sys.executable, str(script), "--url", url, "--csv", str(csv), "--n", "3000"]
    )


def cmd_drift_once() -> int:
    env = {
        **dict(__import__("os").environ),
        "PREDICTIONS_LOG": str(OUT / "logs" / "predictions.jsonl"),
        "REFERENCE_CSV": str(OUT / "data" / "reference.csv"),
        "DRIFT_REPORT_HTML": str(OUT / "reports" / "drift_report.html"),
        "DRIFT_INTERVAL_SEC": "99999",
    }
    code = subprocess.call(
        [sys.executable, "-c", "from drift_check import run_once; run_once()"],
        cwd=OUT / "drift_job",
        env=env,
    )
    report = OUT / "reports" / "drift_report.html"
    if report.is_file():
        print("Report:", report)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 7 helper")
    parser.add_argument(
        "command",
        choices=["prepare", "up", "down", "ps", "smoke", "traffic", "firmware", "drift-once"],
    )
    args = parser.parse_args()

    if args.command == "prepare":
        return cmd_prepare()
    if not OUT.is_dir() and args.command != "prepare":
        print("Run: python lab7-dryrun/prepare_lab7.py", file=sys.stderr)
        return 1
    if args.command == "up":
        return _compose("up", "-d", "--build")
    if args.command == "down":
        return _compose("down")
    if args.command == "ps":
        return _compose("ps")
    if args.command == "smoke":
        return cmd_smoke()
    if args.command == "traffic":
        return cmd_traffic(False)
    if args.command == "firmware":
        return cmd_traffic(True)
    if args.command == "drift-once":
        return cmd_drift_once()
    return 1


if __name__ == "__main__":
    sys.exit(main())
