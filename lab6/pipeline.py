from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import requests
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

LAB6_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB6_DIR.parent
TEMPLATES = LAB6_DIR / "templates"

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

P95_LIMIT_MS = 200.0
ERROR_LIMIT_PCT = 1.0
MAX_F1_DROP = 0.02

V2_CANDIDATES = [(100, 10), (50, 8), (30, 6), (20, 5)]


@dataclass
class LabPaths:
    root: Path
    input_dir: Path
    lab5_dir: Path
    output_dir: Path
    v1_port: int
    v2_port: int
    router_port: int


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class ServerProcs:
    def __init__(self) -> None:
        self._procs: List[subprocess.Popen] = []

    def start_uvicorn(self, app_dir: Path, port: int, env: Optional[Dict[str, str]] = None) -> None:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(app_dir),
            env=run_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(proc)

    def stop_all(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._procs.clear()


def default_paths() -> LabPaths:
    return LabPaths(
        root=REPO_ROOT,
        input_dir=REPO_ROOT / "input",
        lab5_dir=REPO_ROOT / "lab5" / "outputs",
        output_dir=LAB6_DIR / "outputs",
        v1_port=int(os.environ.get("LAB6_V1_PORT", "8001")),
        v2_port=int(os.environ.get("LAB6_V2_PORT", "8002")),
        router_port=int(os.environ.get("LAB6_ROUTER_PORT", "8000")),
    )


def sync_templates(out: Path, paths: LabPaths) -> None:
    for rel in ("model_service", "router", "scripts", "docker-compose.yml"):
        src = TEMPLATES / rel
        dst = out / rel
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    cfg_path = out / "router" / "router_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["v1_url"] = f"http://127.0.0.1:{paths.v1_port}"
    cfg["v2_url"] = f"http://127.0.0.1:{paths.v2_port}"
    cfg["p95_limit_ms"] = P95_LIMIT_MS
    cfg["error_limit_pct"] = ERROR_LIMIT_PCT
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (out / "logs").mkdir(parents=True, exist_ok=True)


def load_datasets(input_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(input_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSV files in {input_dir}")
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


def make_split(df: pd.DataFrame) -> SplitData:
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def onnx_probs(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: X.astype(np.float32)})
    if len(outputs) >= 2 and len(outputs[1].shape) == 2:
        return outputs[1][:, 1]
    probs = outputs[0]
    if probs.ndim == 2:
        return probs[:, 1]
    return probs


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


def export_sklearn_onnx(model: Any, features: List[str], dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    initial_types = [("input", FloatTensorType([None, len(features)]))]
    options = {id(model): {"zipmap": False}}
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=17,
    )
    onnx_path = dest / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    meta = {
        "features": features,
        "target": TARGET,
        "model_type": "random_forest",
        "threshold": 0.5,
        "onnx_input_name": input_name,
    }
    (dest / "features.json").write_text(json.dumps(meta, indent=2))
    return input_name


def write_features_json(dest: Path, meta: Dict[str, Any]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "features.json").write_text(json.dumps(meta, indent=2))


def latency_stats(onnx_path: Path, X: np.ndarray, n: int = 500, threads: Optional[int] = None) -> Tuple[float, float]:
    opts = ort.SessionOptions()
    if threads == 1:
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    times: List[float] = []
    row = X[:1].astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: row})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[int(0.5 * (len(times) - 1))]
    p95 = times[int(0.95 * (len(times) - 1))]
    return p50, p95


def wait_health(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
            last_err = f"status {r.status_code}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(0.3)
    raise RuntimeError(f"Service not healthy at {url}: {last_err}")


def risky_row(split: SplitData) -> Dict[str, Any]:
    idx = split.y_test.sort_values(ascending=False).index[0]
    row = split.X_test.loc[idx]
    payload = {f: float(row[f]) if f != "battery_cycles" else int(row[f]) for f in FEATURES}
    payload["device_id"] = "risky-laptop-demo"
    return payload


def step1(paths: LabPaths, split: SplitData) -> Dict[str, float]:
    v1_dir = paths.output_dir / "models" / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    lab5_onnx = paths.lab5_dir / "model" / "model.onnx"
    lab5_feat = paths.lab5_dir / "model" / "features.json"

    if lab5_onnx.is_file():
        shutil.copy2(lab5_onnx, v1_dir / "model.onnx")
        meta = json.loads(lab5_feat.read_text()) if lab5_feat.is_file() else {}
        meta["version"] = "v1"
        write_features_json(v1_dir, meta)
        print(f"Copied Lab 5 v1 from {lab5_onnx}")
    else:
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)
        rf.fit(split.X_train, split.y_train)
        export_sklearn_onnx(rf, FEATURES, v1_dir)
        meta = json.loads((v1_dir / "features.json").read_text())
        meta["version"] = "v1"
        write_features_json(v1_dir, meta)
        print("Trained fallback v1 RandomForest (no Lab 5 ONNX found)")

    sess = ort.InferenceSession(str(v1_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    probs = onnx_probs(sess, split.X_test.values)
    metrics = classification_metrics(split.y_test.values, probs)
    print("v1 test metrics:", json.dumps(metrics, indent=2))
    print("Step 1 done")
    return metrics


def step2(paths: LabPaths, split: SplitData, v1_f1: float) -> Tuple[Dict[str, float], str]:
    v2_dir = paths.output_dir / "models" / "v2"
    rows = []
    best_any: Optional[Tuple[Any, ...]] = None
    for n_est, depth in V2_CANDIDATES:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        rf.fit(split.X_train, split.y_train)
        sk_prob = rf.predict_proba(split.X_test)[:, 1]
        sk_metrics = classification_metrics(split.y_test.values, sk_prob)
        tmp = paths.output_dir / "models" / f"_tmp_v2_{n_est}_{depth}"
        export_sklearn_onnx(rf, FEATURES, tmp)
        size_kb = (tmp / "model.onnx").stat().st_size / 1024
        sess = ort.InferenceSession(str(tmp / "model.onnx"), providers=["CPUExecutionProvider"])
        onnx_prob = onnx_probs(sess, split.X_test.values)
        max_diff = float(np.max(np.abs(sk_prob - onnx_prob)))
        rows.append((n_est, depth, size_kb, sk_metrics["f1"], sk_metrics["auc_roc"], max_diff, rf, tmp))
        if best_any is None or sk_metrics["f1"] > best_any[3]:
            best_any = rows[-1]

    eligible = [r for r in rows if r[3] >= v1_f1 - MAX_F1_DROP]
    if eligible:
        chosen = min(eligible, key=lambda r: r[2])
        reason = f"smallest model with F1 >= v1_F1 - {MAX_F1_DROP}"
    else:
        chosen = best_any
        reason = "WARNING: no candidate met F1 gate; picked best F1"

    n_est, depth, size_kb, f1, auc, max_diff, rf, tmp = chosen
    if v2_dir.exists():
        shutil.rmtree(v2_dir)
    shutil.copytree(tmp, v2_dir)
    shutil.rmtree(tmp)
    for r in rows:
        p = r[7]
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    meta = json.loads((v2_dir / "features.json").read_text())
    meta["version"] = "v2"
    meta["n_estimators"] = n_est
    meta["max_depth"] = depth
    write_features_json(v2_dir, meta)

    print(f"v2 selected ({n_est}, {depth}): {size_kb:.1f} KB, F1={f1:.4f}, parity max_diff={max_diff:.2e}")
    print("Reason:", reason)
    assert max_diff < 1e-4
    metrics = {"f1": f1, "auc_roc": auc, "size_kb": size_kb}
    print("Step 2 done")
    return metrics, reason


def step3(paths: LabPaths, split: SplitData) -> None:
    fp32_dir = paths.output_dir / "models" / "v2_mlp_fp32"
    int8_dir = paths.output_dir / "models" / "v2_mlp_int8"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(split.X_train, split.y_train)
    export_sklearn_onnx(pipe, FEATURES, fp32_dir)
    meta = json.loads((fp32_dir / "features.json").read_text())
    meta["model_type"] = "mlp_fp32"
    meta["version"] = "mlp_fp32"
    write_features_json(fp32_dir, meta)

    fp32_path = fp32_dir / "model.onnx"
    int8_path = int8_dir / "model.onnx"
    try:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        shutil.copy2(fp32_dir / "features.json", int8_dir / "features.json")
        meta_int8 = json.loads((int8_dir / "features.json").read_text())
        meta_int8["model_type"] = "mlp_int8"
        meta_int8["version"] = "mlp_int8"
        write_features_json(int8_dir, meta_int8)
    except Exception as exc:
        print("INT8 quantize failed (continuing):", exc)

    for label, d in [("FP32", fp32_dir), ("INT8", int8_dir)]:
        if not (d / "model.onnx").is_file():
            continue
        sess = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
        probs = onnx_probs(sess, split.X_test.values)
        m = classification_metrics(split.y_test.values, probs)
        size_kb = (d / "model.onnx").stat().st_size / 1024
        p50, p95 = latency_stats(d / "model.onnx", split.X_test.values, n=200)
        print(f"{label}: size_kb={size_kb:.1f} F1={m['f1']:.4f} p50={p50:.2f}ms p95={p95:.2f}ms")

    if fp32_path.is_file() and int8_path.is_file():
        ratio = fp32_path.stat().st_size / max(int8_path.stat().st_size, 1)
        print(f"Size ratio FP32/INT8: {ratio:.2f}x")
    print("Step 3 done")


def step4(paths: LabPaths, split: SplitData, v1_metrics: Dict[str, float], v2_metrics: Dict[str, float]) -> str:
    entries = []
    model_dirs = [
        ("v1", paths.output_dir / "models" / "v1"),
        ("v2", paths.output_dir / "models" / "v2"),
        ("mlp_fp32", paths.output_dir / "models" / "v2_mlp_fp32"),
        ("mlp_int8", paths.output_dir / "models" / "v2_mlp_int8"),
    ]
    for name, mdir in model_dirs:
        onnx_path = mdir / "model.onnx"
        if not onnx_path.is_file():
            continue
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probs = onnx_probs(sess, split.X_test.values)
        m = classification_metrics(split.y_test.values, probs)
        p50, p95 = latency_stats(onnx_path, split.X_test.values, n=500)
        _, p95_1 = latency_stats(onnx_path, split.X_test.values, n=500, threads=1)
        entries.append(
            {
                "model": name,
                "size_kb": round(onnx_path.stat().st_size / 1024, 2),
                "f1": round(m["f1"], 4),
                "recall": round(m["recall"], 4),
                "auc_roc": round(m["auc_roc"], 4),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p95_1thread_ms": round(p95_1, 3),
            }
        )

    comp_path = paths.output_dir / "comparison.csv"
    pd.DataFrame(entries).to_csv(comp_path, index=False)
    print("Saved", comp_path)

    v2_row = next(e for e in entries if e["model"] == "v2")
    v1_row = next(e for e in entries if e["model"] == "v1")
    accepted = v2_row["f1"] >= v1_row["f1"] - MAX_F1_DROP and v2_row["p95_1thread_ms"] <= P95_LIMIT_MS
    decision = "accepted" if accepted else "rejected"
    reason = f"F1 gate and p95_1thread<={P95_LIMIT_MS}ms"
    print(f"v2 {decision} for edge: {reason}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [e["model"] for e in entries]
    axes[0].bar(names, [e["size_kb"] for e in entries])
    axes[0].set_title("Model size (KB)")
    axes[1].bar(names, [e["p95_1thread_ms"] for e in entries])
    axes[1].set_title("p95 latency (1 thread, ms)")
    fig.tight_layout()
    chart = paths.output_dir / "comparison_charts.png"
    fig.savefig(chart, dpi=120)
    plt.close(fig)
    print("Saved", chart)
    print("Step 4 done")
    return decision


def step5(paths: LabPaths, split: SplitData, servers: ServerProcs) -> None:
    sync_templates(paths.output_dir, paths)
    payload = risky_row(split)

    ms_dir = paths.output_dir / "model_service"
    servers.start_uvicorn(ms_dir, paths.v1_port, {"MODEL_DIR": str(paths.output_dir / "models" / "v1")})
    servers.start_uvicorn(ms_dir, paths.v2_port, {"MODEL_DIR": str(paths.output_dir / "models" / "v2")})

    wait_health(f"http://127.0.0.1:{paths.v1_port}/health")
    wait_health(f"http://127.0.0.1:{paths.v2_port}/health")

    for port, label in [(paths.v1_port, "v1"), (paths.v2_port, "v2")]:
        r = requests.post(f"http://127.0.0.1:{port}/predict", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"{label} predict:", json.dumps(data, indent=2))
        assert data.get("version") in ("v1", "v2")
    print("Step 5 done")


def step6(paths: LabPaths, servers: ServerProcs) -> None:
    router_dir = paths.output_dir / "router"
    servers.start_uvicorn(
        router_dir,
        paths.router_port,
        {
            "ROUTER_CONFIG": str(router_dir / "router_config.json"),
            "LOG_DIR": str(paths.output_dir / "logs"),
        },
    )
    wait_health(f"http://127.0.0.1:{paths.router_port}/health")
    m = requests.get(f"http://127.0.0.1:{paths.router_port}/metrics", timeout=10).json()
    print("Router metrics:", json.dumps(m, indent=2))
    assert m.get("v2_percent", 0) == 0
    print("Step 6 done")


def step7(paths: LabPaths, split: SplitData, servers: ServerProcs) -> None:
    csv_path = paths.input_dir / "device_telemetry.csv"
    script = paths.output_dir / "scripts" / "send_traffic.py"
    router_base = f"http://127.0.0.1:{paths.router_port}"

    for pct in (5, 25, 100):
        requests.post(f"{router_base}/config", json={"v2_percent": pct}, timeout=10).raise_for_status()
        subprocess.run(
            [sys.executable, str(script), "--url", f"{router_base}/predict", "--csv", str(csv_path), "--n", "1000"],
            check=True,
        )
        metrics = requests.get(f"{router_base}/metrics", timeout=10).json()
        v2 = metrics["v2"]
        ok = v2["p95_ms"] <= P95_LIMIT_MS and v2["error_rate_pct"] <= ERROR_LIMIT_PCT
        print(f"Rollout {pct}%: v2 p95={v2['p95_ms']}ms errors={v2['error_rate_pct']}% -> {'PASS' if ok else 'FAIL'}")

    # Bad v2 release: restart v2 with extra delay
    servers.stop_all()
    time.sleep(1)
    ms_dir = paths.output_dir / "model_service"
    servers.start_uvicorn(ms_dir, paths.v1_port, {"MODEL_DIR": str(paths.output_dir / "models" / "v1")})
    servers.start_uvicorn(
        ms_dir,
        paths.v2_port,
        {"MODEL_DIR": str(paths.output_dir / "models" / "v2"), "EXTRA_DELAY_MS": "300"},
    )
    router_dir = paths.output_dir / "router"
    servers.start_uvicorn(
        router_dir,
        paths.router_port,
        {
            "ROUTER_CONFIG": str(router_dir / "router_config.json"),
            "LOG_DIR": str(paths.output_dir / "logs"),
        },
    )
    wait_health(f"http://127.0.0.1:{paths.router_port}/health")
    requests.post(f"{router_base}/config", json={"v2_percent": 25}, timeout=10).raise_for_status()
    subprocess.run(
        [sys.executable, str(script), "--url", f"{router_base}/predict", "--csv", str(csv_path), "--n", "1000"],
        check=True,
    )
    metrics = requests.get(f"{router_base}/metrics", timeout=10).json()
    print("After bad v2 metrics:", json.dumps(metrics, indent=2))
    rollback_log = paths.output_dir / "logs" / "rollback_events.jsonl"
    if rollback_log.is_file():
        print("Rollback events (last line):", rollback_log.read_text().strip().splitlines()[-1])
    assert metrics.get("v2_percent", -1) == 0 or metrics.get("rolled_back") is True

    # Restore normal v2
    servers.stop_all()
    time.sleep(1)
    step5(paths, split, servers)
    step6(paths, servers)
    print("Step 7 done")


def step8(paths: LabPaths, servers: ServerProcs) -> None:
    zip_path = paths.output_dir / "lab6_app.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(paths.output_dir):
            root_path = Path(root)
            if root_path.name == "__pycache__":
                continue
            for fname in files:
                fp = root_path / fname
                if fp == zip_path:
                    continue
                arc = fp.relative_to(paths.output_dir)
                zf.write(fp, arcname=str(arc))
    print("Saved", zip_path)
    servers.stop_all()
    print("Step 8 done — servers stopped")
    print(
        "Laptop: unzip lab6_app.zip && cd lab6/outputs && docker compose up -d --build"
    )


def run_pipeline(steps: Optional[List[int]] = None) -> None:
    paths = default_paths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_datasets(paths.input_dir)
    split = make_split(df)
    print(f"Loaded {len(df)} rows; failure rate {df[TARGET].mean():.2%}")

    all_steps = steps or list(range(1, 9))
    servers = ServerProcs()
    v1_metrics: Dict[str, float] = {}
    v2_metrics: Dict[str, float] = {}

    try:
        if 1 in all_steps:
            v1_metrics = step1(paths, split)
        if 2 in all_steps:
            v1_f1 = v1_metrics.get("f1")
            if v1_f1 is None:
                sess = ort.InferenceSession(
                    str(paths.output_dir / "models" / "v1" / "model.onnx"),
                    providers=["CPUExecutionProvider"],
                )
                probs = onnx_probs(sess, split.X_test.values)
                v1_f1 = classification_metrics(split.y_test.values, probs)["f1"]
            v2_metrics, _ = step2(paths, split, v1_f1)
        if 3 in all_steps:
            step3(paths, split)
        if 4 in all_steps:
            if not v1_metrics:
                v1_metrics = {"f1": 0}
            if not v2_metrics:
                v2_metrics = {"f1": 0}
            step4(paths, split, v1_metrics, v2_metrics)
        if 5 in all_steps:
            step5(paths, split, servers)
        if 6 in all_steps:
            if 5 not in all_steps:
                step5(paths, split, servers)
            step6(paths, servers)
        if 7 in all_steps:
            if 6 not in all_steps:
                step5(paths, split, servers)
                step6(paths, servers)
            step7(paths, split, servers)
        if 8 in all_steps:
            step8(paths, servers)
    finally:
        servers.stop_all()
