#!/usr/bin/env bash
# Local setup for data-drift-ML-PSI (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name lab5-local --display-name "Python 3 (lab5-local)"
echo "Done. Activate: source .venv/bin/activate"
echo ""
echo "Project layout:"
echo "  input/           shared training CSVs"
echo "  lab5/            failure model notebook + outputs/"
echo "  lab6/            edge canary pipeline + outputs/"
echo "  lab7-dryrun/     monitoring / drift Docker stack + outputs/"
echo ""
echo "Lab 5:  jupyter notebook lab5/lab5_failure_model.ipynb"
echo "        python lab5/run_notebook.py"
echo "Lab 6:  python lab6/run_lab6.py"
echo "Lab 7:  python lab7-dryrun/prepare_lab7.py"
echo "        cd lab7-dryrun/outputs && docker compose up -d --build"
