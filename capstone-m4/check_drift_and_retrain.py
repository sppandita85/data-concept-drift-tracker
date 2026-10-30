#!/usr/bin/env python3
"""
Capstone M4 — if drift (cpu_temp_avg) >= threshold, run retrain_trigger (Steps 11–12).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import requests

CAPSTONE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAPSTONE_DIR.parent
LAB7_OUT = REPO_ROOT / "lab7-dryrun" / "outputs"
DRIFT_METRICS_URL = "http://127.0.0.1:9100/metrics"
DEFAULT_THRESHOLD = 0.25
FEATURE = "cpu_temp_avg"


def _score_from_prometheus(text: str, feature: str) -> float | None:
    pat = re.compile(
        rf'^feature_drift_score{{feature="{re.escape(feature)}"}}\s+([0-9.eE+-]+)\s*$'
    )
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = pat.match(line.strip())
        if m:
            return float(m.group(1))
    return None


def _score_from_local_drift() -> float | None:
    log = LAB7_OUT / "logs" / "predictions.jsonl"
    ref = LAB7_OUT / "data" / "reference.csv"
    if not log.is_file() or not ref.is_file():
        return None
    import os

    env = {
        **os.environ,
        "PREDICTIONS_LOG": str(log),
        "REFERENCE_CSV": str(ref),
        "DRIFT_REPORT_HTML": str(LAB7_OUT / "reports" / "drift_report.html"),
    }
    code = subprocess.call(
        [sys.executable, "-c", "from drift_check import run_once; run_once()"],
        cwd=LAB7_OUT / "drift_job",
        env=env,
    )
    if code != 0:
        return None
    try:
        text = requests.get(DRIFT_METRICS_URL, timeout=5).text
        return _score_from_prometheus(text, FEATURE)
    except requests.RequestException:
        return None


def read_drift_score(feature: str = FEATURE) -> float | None:
    try:
        r = requests.get(DRIFT_METRICS_URL, timeout=10)
        r.raise_for_status()
        score = _score_from_prometheus(r.text, feature)
        if score is not None:
            return score
    except requests.RequestException:
        pass
    return _score_from_local_drift()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--feature", default=FEATURE)
    p.add_argument(
        "--force",
        action="store_true",
        help="Run retrain even if drift is below threshold",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-docker-restart", action="store_true")
    args = p.parse_args()

    score = read_drift_score(args.feature)
    if score is None:
        print("Could not read drift score (is drift container up on :9100?)", file=sys.stderr)
        if not args.force:
            return 2

    print(f"Drift score {args.feature}: {score if score is not None else 'unknown'}")
    if not args.force and score is not None and score < args.threshold:
        print(f"Below threshold {args.threshold}; no retrain.")
        return 0

    if args.dry_run:
        print("Dry run — would invoke retrain_trigger.py")
        return 0

    cmd = [
        sys.executable,
        str(CAPSTONE_DIR / "retrain_trigger.py"),
        "--reason",
        f"drift_{args.feature}_gte_{args.threshold}",
    ]
    if score is not None:
        cmd.extend(["--drift-score", str(score)])
    if args.skip_docker_restart:
        cmd.append("--skip-docker-restart")

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
