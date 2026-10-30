#!/usr/bin/env bash
# Capstone M4 Steps 13–14 — end-to-end demo (Lab 7 stack must be runnable).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
LAB7="$REPO_ROOT/lab7-dryrun"
CAPSTONE="$REPO_ROOT/capstone-m4"
OUT="$LAB7/outputs"
WAIT_DRIFT="${WAIT_DRIFT:-90}"
FRESH="${FRESH:-0}"

if [[ "$FRESH" == "1" ]]; then
  "$PYTHON" "$LAB7/prepare_lab7.py"
fi

if [[ ! -d "$OUT" ]]; then
  echo "Run: python lab7-dryrun/prepare_lab7.py (or FRESH=1 $0)" >&2
  exit 1
fi

echo "=== Docker Compose up ==="
(cd "$OUT" && docker compose up -d --build)
sleep 5

echo "=== Step 6 — normal traffic ==="
"$PYTHON" "$LAB7/run_lab7.py" traffic

echo "=== Wait for drift job ($WAIT_DRIFT s) ==="
sleep "$WAIT_DRIFT"

echo "=== Step 9 — firmware (+8°C) traffic ==="
"$PYTHON" "$LAB7/run_lab7.py" firmware

echo "=== Wait for drift alert window ($WAIT_DRIFT s) ==="
sleep "$WAIT_DRIFT"

echo "=== Steps 11–12 — drift gate + retrain + baseline refresh ==="
"$PYTHON" "$CAPSTONE/check_drift_and_retrain.py" || "$PYTHON" "$CAPSTONE/check_drift_and_retrain.py" --force

echo "=== Step 13 — recovery: normal traffic (flush log window) ==="
"$PYTHON" "$LAB7/run_lab7.py" traffic
"$PYTHON" "$LAB7/run_lab7.py" traffic

echo "=== Wait for drift recovery ($WAIT_DRIFT s) ==="
sleep "$WAIT_DRIFT"
"$PYTHON" "$LAB7/run_lab7.py" drift-once

echo ""
echo "=== Open for presentation ==="
echo "  Grafana:    http://localhost:3000  (admin / admin)"
echo "  API docs:   http://localhost:8088/docs"
echo "  Evidently:  file://$OUT/reports/drift_report.html"
echo "  Retrain log: $CAPSTONE/logs/retrain_events.jsonl"
echo ""
echo "Capture: Grafana before/after firmware, fired alert, after recovery, Evidently HTML."
