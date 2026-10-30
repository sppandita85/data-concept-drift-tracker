#!/usr/bin/env python3
"""Capstone M4 Step 12 only — refresh lab7 reference.csv from input/*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

# Reuse prepare logic via retrain_trigger helper
CAPSTONE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CAPSTONE_DIR))

from retrain_trigger import refresh_reference_csv  # noqa: E402


def main() -> int:
    path = refresh_reference_csv()
    print("Wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
