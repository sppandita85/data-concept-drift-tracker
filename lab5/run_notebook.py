#!/usr/bin/env python3
"""Execute lab5_failure_model.ipynb from the project root."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

LAB5_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB5_DIR.parent


def main() -> int:
    nb_path = LAB5_DIR / "lab5_failure_model.ipynb"
    nb = nbformat.read(nb_path.open(), as_version=4)
    out_path = LAB5_DIR / "lab5_failure_model_executed.ipynb"

    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="lab5-local",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    print("Running notebook from", REPO_ROOT)
    client.execute()
    out_path.write_text(nbformat.writes(nb))
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
