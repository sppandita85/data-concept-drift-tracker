"""Minimal Lab-5-style train + ONNX export for capstone retrain demo."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

FEATURES = [
    "cpu_temp_avg",
    "battery_cycles",
    "device_age_months",
    "disk_errors",
    "crash_count_30d",
    "battery_health_pct",
]
TARGET = "failed_next_30d"
RANDOM_STATE = 42


def load_labeled_csv(input_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(input_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSV in {input_dir}")
    df = pd.concat(frames, ignore_index=True)
    required = ["device_id", *FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.drop_duplicates()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    return df


def train_and_export(input_dir: Path, model_dir: Path, version: str) -> Dict[str, Any]:
    df = load_labeled_csv(input_dir)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": float(f1_score(y_test, (proba >= 0.5).astype(int))),
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    initial_types = [("input", FloatTensorType([None, len(FEATURES)]))]
    options = {id(model): {"zipmap": False}}
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=17,
    )
    onnx_path = model_dir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    meta = {
        "features": FEATURES,
        "target": TARGET,
        "model_type": "random_forest",
        "threshold": 0.5,
        "onnx_input_name": input_name,
        "version": version,
    }
    (model_dir / "features.json").write_text(json.dumps(meta, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "onnx_path": str(onnx_path), "meta": meta}


def copy_model_tree(src_dir: Path, *dest_dirs: Path) -> None:
    for dest in dest_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("model.onnx", "features.json"):
            shutil.copy2(src_dir / name, dest / name)


def default_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"capstone-retrain-{ts}"
