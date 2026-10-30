#!/usr/bin/env python3
"""Stage Lab 7 stack under lab7-dryrun/outputs/ (API + reference data + compose files)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

LAB7_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB7_DIR.parent
TEMPLATES = LAB7_DIR / "templates"
OUT = LAB7_DIR / "outputs"


def _concat_training_csv() -> None:
    inp = REPO_ROOT / "input"
    frames = []
    for p in sorted(inp.glob("*.csv")):
        frames.append(p.read_text())
    if not frames:
        raise FileNotFoundError(f"No CSV in {inp}")
    header = frames[0].splitlines()[0]
    lines = [header]
    for text in frames:
        lines.extend(text.splitlines()[1:])
    ref = OUT / "data" / "reference.csv"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text("\n".join(lines) + "\n")
    print("Wrote", ref, f"({len(lines)-1} rows)")


def _copy_model() -> None:
    api = OUT / "api"
    api.mkdir(parents=True, exist_ok=True)
    candidates = [
        REPO_ROOT / "lab5" / "outputs" / "app",
        REPO_ROOT / "lab5" / "outputs" / "model",
    ]
    src_dir = next((c for c in candidates if (c / "model.onnx").is_file()), None)
    if src_dir is None:
        raise FileNotFoundError("Run Lab 5 first — need lab5/outputs/app/model.onnx")
    shutil.copy2(src_dir / "model.onnx", api / "model.onnx")
    meta = json.loads((src_dir / "features.json").read_text())
    meta.setdefault("version", "lab5-v1")
    (api / "features.json").write_text(json.dumps(meta, indent=2))
    print("Copied model from", src_dir)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(TEMPLATES, OUT)
    _copy_model()
    _concat_training_csv()
    for d in ("logs", "reports"):
        (OUT / d).mkdir(exist_ok=True)
    print("\nLab 7 staged at", OUT)
    print("  cd lab7-dryrun/outputs && docker compose up -d --build")
    print("  Grafana: http://localhost:3000 (admin / admin)")
    print("  API docs: http://localhost:8088/docs  (override LAB7_API_PORT in compose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
