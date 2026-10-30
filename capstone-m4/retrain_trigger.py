#!/usr/bin/env python3
"""
Capstone M4 Step 11 — demo retrain: train on labeled input/*.csv, export ONNX, deploy to Lab 7 API.

Does not train from predictions.jsonl (no labels there). See README for the teaching story.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from training import copy_model_tree, default_version, train_and_export

CAPSTONE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAPSTONE_DIR.parent
INPUT_DIR = REPO_ROOT / "input"
LAB5_MODEL = REPO_ROOT / "lab5" / "outputs" / "model"
LAB5_APP = REPO_ROOT / "lab5" / "outputs" / "app"
LAB7_OUT = REPO_ROOT / "lab7-dryrun" / "outputs"
LAB7_API = LAB7_OUT / "api"
REFERENCE_CSV = LAB7_OUT / "data" / "reference.csv"
RETRAIN_TS_FILE = LAB7_OUT / "data" / "retrain_last_success_unixtime.txt"
LOG_DIR = CAPSTONE_DIR / "logs"
EVENTS_LOG = LOG_DIR / "retrain_events.jsonl"


def refresh_reference_csv() -> Path:
    """Step 12 — rewrite reference baseline from input/*.csv (same as prepare_lab7)."""
    frames = []
    for p in sorted(INPUT_DIR.glob("*.csv")):
        frames.append(p.read_text())
    if not frames:
        raise FileNotFoundError(f"No CSV in {INPUT_DIR}")
    header = frames[0].splitlines()[0]
    lines = [header]
    for text in frames:
        lines.extend(text.splitlines()[1:])
    REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_CSV.write_text("\n".join(lines) + "\n")
    return REFERENCE_CSV


def append_event(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def restart_lab7_api() -> None:
    if not (LAB7_OUT / "docker-compose.yml").is_file():
        print("Skip docker rebuild: lab7-dryrun/outputs not staged", file=sys.stderr)
        return
    # model.onnx is COPY'd into the image at build time (not a runtime volume)
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build", "api"],
        cwd=LAB7_OUT,
        check=False,
    )


def run_retrain(
    *,
    reason: str,
    drift_score: float | None,
    version: str | None,
    skip_docker: bool,
    refresh_baseline: bool,
) -> dict:
    ver = version or default_version()
    t0 = time.perf_counter()
    result = train_and_export(INPUT_DIR, LAB5_MODEL, ver)
    copy_model_tree(LAB5_MODEL, LAB5_APP, LAB7_API)

    ref_path = None
    if refresh_baseline:
        ref_path = str(refresh_reference_csv())

    ts = time.time()
    RETRAIN_TS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_TS_FILE.write_text(f"{ts:.3f}\n")

    elapsed = time.perf_counter() - t0
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "drift_score": drift_score,
        "model_version": ver,
        "lab7_api_dir": str(LAB7_API),
        "reference_csv": ref_path,
        "metrics": result["metrics"],
        "elapsed_sec": round(elapsed, 2),
    }
    append_event(event)

    if not skip_docker:
        restart_lab7_api()

    return event


def main() -> int:
    p = argparse.ArgumentParser(description="Capstone M4 demo retrain trigger")
    p.add_argument("--reason", default="manual")
    p.add_argument("--drift-score", type=float, default=None)
    p.add_argument("--version", default=None, help="model_version label in features.json")
    p.add_argument("--skip-docker-restart", action="store_true")
    p.add_argument(
        "--no-refresh-baseline",
        action="store_true",
        help="Skip Step 12 reference.csv refresh",
    )
    args = p.parse_args()

    try:
        event = run_retrain(
            reason=args.reason,
            drift_score=args.drift_score,
            version=args.version,
            skip_docker=args.skip_docker_restart,
            refresh_baseline=not args.no_refresh_baseline,
        )
    except Exception as exc:
        print("Retrain failed:", exc, file=sys.stderr)
        return 1

    print(json.dumps(event, indent=2))
    print(f"\nLogged → {EVENTS_LOG}")
    if LAB7_OUT.is_dir():
        print("API model dir:", LAB7_API)
    return 0


if __name__ == "__main__":
    sys.exit(main())
