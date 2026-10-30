#!/usr/bin/env python3
"""Run Lab 6 locally (edge model + canary router). No Colab required."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 6 local pipeline")
    parser.add_argument(
        "--steps",
        type=str,
        default="1-8",
        help="Step numbers, e.g. 1-4 or 1,3,5",
    )
    args = parser.parse_args()

    if "-" in args.steps and "," not in args.steps:
        lo, hi = args.steps.split("-", 1)
        steps = list(range(int(lo), int(hi) + 1))
    else:
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    print("Running Lab 6 steps:", steps)
    print("Ports: router LAB6_ROUTER_PORT (8000), v1 LAB6_V1_PORT (8001), v2 LAB6_V2_PORT (8002)")
    run_pipeline(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
