#!/usr/bin/env python3
"""Build lab5_failure_model.ipynb for local execution (no Colab)."""
from __future__ import annotations

import json
from pathlib import Path

LAB5_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB5_DIR.parent
ASSETS = LAB5_DIR / "templates"


def _read_asset(rel: str, fallback: str = "") -> str:
    path = ASSETS / rel
    if path.is_file():
        return path.read_text()
    return fallback


MAIN_PY = _read_asset("app/main.py")
APP_REQUIREMENTS = _read_asset("app/requirements.txt")
DOCKERFILE = _read_asset("app/Dockerfile")
DOCKERIGNORE = _read_asset("app/.dockerignore")
CALL_API = _read_asset("scripts/call_api.py")
LATENCY_TEST = _read_asset("scripts/latency_test.py")
COMPARE_LOAD = _read_asset("scripts/compare_load.py")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells: list[dict] = []

cells.append(
    md(
        """# Lab 5: Failure Model — Train, Export ONNX, Serve with FastAPI

**Local project** — run cells in order from the repo root (`lab5-day3-lenovo/`).  
Data: `input/*.csv` (repo root) · Artifacts: `lab5/outputs/`
"""
    )
)

# --- Step 1 ---
cells.append(
    md(
        """## Step 1: Load and check the data

We load all CSV files from `input/` and inspect shape, types, and class balance.  
Cleaning fixes missing values and keeps battery health in a valid range before modeling.
"""
    )
)

cells.append(
    code(
        """import subprocess
import sys
from pathlib import Path

# One-time from terminal: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
def _ensure_deps() -> None:
    try:
        import sklearn  # noqa: F401
        import xgboost  # noqa: F401
        import onnxruntime  # noqa: F401
        import fastapi  # noqa: F401
        print("Dependencies OK (skip pip install)")
        return
    except ImportError:
        pass
    root = Path.cwd()
    req = root / "requirements.txt"
    if not req.is_file():
        raise FileNotFoundError(f"Missing {req} — run notebook from project root")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        cwd=str(root),
    )


_ensure_deps()
"""
    )
)

cells.append(
    code(
        """import os
import glob
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    from IPython.display import display
except ImportError:
    display = print


def find_project_root() -> Path:
    here = Path.cwd().resolve()
    if (here / "input").is_dir() and (here / "requirements.txt").is_file():
        return here
    for parent in [here, *here.parents]:
        if (parent / "input").is_dir():
            return parent
    return here


PROJECT_ROOT = find_project_root()
INPUT_DIR = str(PROJECT_ROOT / "input")
OUTPUT_DIR = str(PROJECT_ROOT / "lab5" / "outputs")

FEATURES = [
    "cpu_temp_avg",
    "battery_cycles",
    "device_age_months",
    "disk_errors",
    "crash_count_30d",
    "battery_health_pct",
]
TARGET = "failed_next_30d"
ID_COL = "device_id"
RANDOM_STATE = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "model").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "app").mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR, "scripts").mkdir(parents=True, exist_ok=True)


def write_lab_file(rel_path: str, content: str) -> Path:
    dest = Path(OUTPUT_DIR) / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content.rstrip("\\n") + "\\n")
    print(f"Wrote {dest}")
    return dest


print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
"""
    )
)

cells.append(
    code(
        """REQUIRED_COLS = [ID_COL] + FEATURES + [TARGET]
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files in {INPUT_DIR}. Add CSVs under input/ and re-run."
    )

parts = [pd.read_csv(p) for p in csv_files]
df = pd.concat(parts, ignore_index=True)
print("Loaded:", [os.path.basename(p) for p in csv_files])
print("df.shape:", df.shape)
display(df.head())
print("\\nDtypes:\\n", df.dtypes)
print("\\nMissing:\\n", df.isna().sum())
print(f"Duplicate {ID_COL} rows:", df[ID_COL].duplicated().sum())
"""
    )
)

cells.append(
    code(
        """counts = df[TARGET].value_counts().sort_index()
pcts = (counts / len(df) * 100).round(2)
print("Class balance:")
for label in counts.index:
    print(f"  {label}: {counts[label]} ({pcts[label]}%)")

plt.figure(figsize=(4, 3))
counts.plot(kind="bar", color=["#4C78A8", "#E45756"])
plt.title("Target class balance")
plt.xlabel(TARGET)
plt.ylabel("count")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """before = len(df)
df = df.drop_duplicates()
print("Dropped exact duplicate rows:", before - len(df))

filled = {}
for col in FEATURES:
    n_missing = int(df[col].isna().sum())
    if n_missing:
        df[col] = df[col].fillna(df[col].median())
        filled[col] = n_missing
print("Filled missing (per column):", filled or "none")

bad = (df["battery_health_pct"] < 0) | (df["battery_health_pct"] > 100)
n_clip = int(bad.sum())
df["battery_health_pct"] = df["battery_health_pct"].clip(0, 100)
print("battery_health_pct clipped to [0,100]:", n_clip)
"""
    )
)

cells.append(
    code(
        """print(df[FEATURES].describe())

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for ax, col in zip(axes.ravel(), FEATURES):
    ax.hist(df[col], bins=30, color="#72B7B2", edgecolor="white")
    ax.set_title(col)
plt.suptitle("Feature distributions")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """failure_rate = df[TARGET].mean() * 100
print(
    f"Failure rate is {failure_rate:.2f}% - this is imbalanced, so we will use "
    "precision, recall, F1 and AUC-ROC, not just accuracy."
)
print("Step 1 done")
"""
    )
)

# --- Step 2 ---
cells.append(
    md(
        """## Step 2: Train random forest and XGBoost, pick the winner

Two tree models on an 80/20 stratified split; best F1 (AUC tie-break) is exported and served.
"""
    )
)

cells.append(
    code(
        """from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
"""
    )
)

cells.append(md("Tree models use raw features — **no scaling** for Random Forest or XGBoost."))

cells.append(
    code(
        """rf_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
)
rf_model.fit(X_train, y_train)

neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=neg / max(pos, 1),
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)
xgb_model.fit(X_train, y_train)
"""
    )
)

cells.append(
    code(
        """def evaluate_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, proba),
    }
    print(f"\\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return metrics, proba

rf_metrics, rf_proba = evaluate_model("RandomForest", rf_model, X_test, y_test)
xgb_metrics, xgb_proba = evaluate_model("XGBoost", xgb_model, X_test, y_test)
"""
    )
)

cells.append(
    code(
        """comparison = pd.DataFrame([rf_metrics, xgb_metrics]).set_index("model")
display(comparison.round(4))

plt.figure(figsize=(6, 4))
for name, proba in [("RandomForest", rf_proba), ("XGBoost", xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC — test set")
plt.legend()
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """winner_metrics = max([rf_metrics, xgb_metrics], key=lambda r: (r["f1"], r["auc_roc"]))
winner_name = winner_metrics["model"]
if winner_name == "RandomForest":
    winner_model, winner_key = rf_model, "random_forest"
else:
    winner_model, winner_key = xgb_model, "xgboost"
print(
    f"Winner: {winner_name} — F1={winner_metrics['f1']:.4f}, "
    f"AUC={winner_metrics['auc_roc']:.4f}"
)
"""
    )
)

cells.append(
    code(
        """order = np.argsort(winner_model.feature_importances_)
plt.figure(figsize=(6, 4))
plt.barh(np.array(FEATURES)[order], winner_model.feature_importances_[order])
plt.title(f"Feature importance — {winner_name}")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """metrics_path = Path(OUTPUT_DIR) / "model" / "metrics.json"
payload = {
    "random_forest": {k: v for k, v in rf_metrics.items() if k != "model"},
    "xgboost": {k: v for k, v in xgb_metrics.items() if k != "model"},
    "winner": winner_key,
    "winner_display_name": winner_name,
}
metrics_path.write_text(json.dumps(payload, indent=2))
print("Saved", metrics_path)
print("Step 2 done")
"""
    )
)

cells.append(md("We trained two models; we export and serve **only the winner**."))

# --- Step 3 ---
cells.append(
    md(
        """## Step 3: Export the winner to ONNX + parity check

ONNX lets the API run without scikit-learn or XGBoost. We verify probabilities match on the test set.
"""
    )
)

cells.append(
    code(
        """import copy

import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as XgbFloatTensorType


def export_winner_to_onnx(model, winner_key, features):
    if winner_key == "random_forest":
        initial_types = [("input", SklearnFloatTensorType([None, len(features)]))]
        options = {id(model): {"zipmap": False}}
        return convert_sklearn(model, initial_types=initial_types, options=options)
    model = copy.deepcopy(model)
    initial_types = [("input", XgbFloatTensorType([None, len(features)]))]
    model.get_booster().feature_names = [f"f{i}" for i in range(len(features))]
    return convert_xgboost(model, initial_types=initial_types)


onnx_model = export_winner_to_onnx(winner_model, winner_key, FEATURES)
onnx_path = Path(OUTPUT_DIR) / "model" / "model.onnx"
onnx_path.write_bytes(onnx_model.SerializeToString())
size_kb = onnx_path.stat().st_size / 1024
print(f"Saved {onnx_path} ({size_kb:.1f} KB)")
"""
    )
)

cells.append(
    code(
        """sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_input_name = sess.get_inputs()[0].name
features_json_path = Path(OUTPUT_DIR) / "model" / "features.json"
features_json_path.write_text(
    json.dumps(
        {
            "features": FEATURES,
            "target": TARGET,
            "model_type": winner_key,
            "threshold": 0.5,
            "onnx_input_name": onnx_input_name,
        },
        indent=2,
    )
)
print("Saved", features_json_path)
"""
    )
)

cells.append(
    code(
        """X_test_np = X_test[FEATURES].values.astype(np.float32)
sk_proba = winner_model.predict_proba(X_test)[:, 1]
outputs = sess.run(None, {onnx_input_name: X_test_np})
onnx_proba = outputs[1][:, 1] if len(outputs) >= 2 and len(outputs[1].shape) == 2 else outputs[0][:, 1]

max_diff = float(np.max(np.abs(sk_proba - onnx_proba)))
match_pct = ((sk_proba >= 0.5) == (onnx_proba >= 0.5)).mean() * 100
print("Max |prob diff|:", max_diff)
assert max_diff < 1e-4
print(f"Label match: {match_pct:.1f}%")
assert match_pct == 100.0
print("Step 3 done")
"""
    )
)

cells.append(
    md("ONNX is an open serving format — the API does **not** need scikit-learn or XGBoost.")
)

# --- Step 4 ---
cells.append(
    md(
        """## Step 4: Write the FastAPI app

Create `lab5/outputs/app/` with `main.py`, dependencies, and copied model files.
"""
    )
)

cells.append(code(f"write_lab_file('app/main.py', {json.dumps(MAIN_PY)})\n"))
cells.append(code(f"write_lab_file('app/requirements.txt', {json.dumps(APP_REQUIREMENTS)})\n"))

cells.append(
    code(
        """shutil.copy2(onnx_path, Path(OUTPUT_DIR) / "app" / "model.onnx")
shutil.copy2(features_json_path, Path(OUTPUT_DIR) / "app" / "features.json")

for root, _, files in os.walk(Path(OUTPUT_DIR) / "app"):
    level = root.replace(str(Path(OUTPUT_DIR) / "app"), "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in sorted(files):
        print(f"{'  ' * (level + 1)}{f}")
print("Step 4 done")
"""
    )
)

# --- Step 5 ---
cells.append(
    md(
        """## Step 5: Run the API locally and test it

Start uvicorn in the background, then call `/health`, `/predict`, and `/predict/batch`.
"""
    )
)

cells.append(
    code(
        """APP_DIR = Path(OUTPUT_DIR) / "app"
server_log = APP_DIR / "server.log"
log_file = open(server_log, "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(APP_DIR),
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

health_url = "http://127.0.0.1:8000/health"
ready = False
for _ in range(40):
    try:
        if requests.get(health_url, timeout=1).status_code == 200:
            ready = True
            break
    except requests.RequestException:
        pass
    time.sleep(0.5)
if not ready:
    raise RuntimeError("API not ready in 20s — see lab5/outputs/app/server.log")
print("Server ready:", health_url)
"""
    )
)

cells.append(
    code(
        """print(requests.get(health_url, timeout=5).json())
healthy = {
    "device_id": "demo-healthy",
    "cpu_temp_avg": 55,
    "battery_cycles": 100,
    "device_age_months": 8,
    "disk_errors": 0,
    "crash_count_30d": 0,
    "battery_health_pct": 95,
}
risky = {
    "device_id": "demo-risky",
    "cpu_temp_avg": 85,
    "battery_cycles": 650,
    "device_age_months": 50,
    "disk_errors": 4,
    "crash_count_30d": 3,
    "battery_health_pct": 68,
}
r1 = requests.post("http://127.0.0.1:8000/predict", json=healthy, timeout=10).json()
r2 = requests.post("http://127.0.0.1:8000/predict", json=risky, timeout=10).json()
print("Healthy:", r1)
print("Risky:", r2)
print("Risky > healthy:", r2["failure_risk"] > r1["failure_risk"])
"""
    )
)

cells.append(
    code(
        """payload = X_test.iloc[:20][FEATURES].to_dict(orient="records")
api_batch = requests.post(
    "http://127.0.0.1:8000/predict/batch", json=payload, timeout=30
).json()
api_probs = np.array([x["failure_risk"] for x in api_batch])
print("Batch matches ONNX:", np.allclose(api_probs, onnx_proba[:20], atol=1e-4))
bad = requests.post(
    "http://127.0.0.1:8000/predict", json={**healthy, "battery_health_pct": 150}, timeout=10
)
print("Bad request:", bad.status_code)
assert bad.status_code == 422
print("Step 5 done")
"""
    )
)

cells.append(
    md(
        """Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (keep the server running for Step 8).
"""
    )
)

# --- Step 6 ---
cells.append(
    md(
        """## Step 6: Docker files and zip bundle

Write Dockerfile and helper scripts (`call_api.py`, `latency_test.py`, `compare_load.py`), then zip `app/` + `scripts/` for deployment.
"""
    )
)

cells.append(code(f"write_lab_file('app/Dockerfile', {json.dumps(DOCKERFILE)})\n"))
cells.append(code(f"write_lab_file('app/.dockerignore', {json.dumps(DOCKERIGNORE)})\n"))
cells.append(code(f"write_lab_file('scripts/call_api.py', {json.dumps(CALL_API)})\n"))
cells.append(code(f"write_lab_file('scripts/latency_test.py', {json.dumps(LATENCY_TEST)})\n"))
cells.append(code(f"write_lab_file('scripts/compare_load.py', {json.dumps(COMPARE_LOAD)})\n"))

cells.append(
    code(
        """zip_path = Path(OUTPUT_DIR) / "lab5_app.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ("app", "scripts"):
        base = Path(OUTPUT_DIR) / folder
        for file in base.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=str(Path(folder) / file.relative_to(base)))
print("Created", zip_path, f"({zip_path.stat().st_size / 1024:.1f} KB)")
print("Step 6 done")
"""
    )
)

cells.append(
    md(
        """### Run with Docker (from repo root after unzipping or using `lab5/outputs/`)

```bash
cd lab5/outputs/app
docker build -t laptop-failure-api:v1 .
docker run -d -p 8000:8000 --name failure-api laptop-failure-api:v1
docker ps
docker logs failure-api
```
"""
    )
)

# --- Step 7 ---
cells.append(
    md(
        """## Step 7: Call the container

With the container running:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"device_id":"PF00007","cpu_temp_avg":73.7,"battery_cycles":611,"device_age_months":42,"disk_errors":4,"crash_count_30d":2,"battery_health_pct":71.0}'

python lab5/outputs/scripts/call_api.py --url http://localhost:8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).

**Expected `/predict` JSON:** `device_id`, `failure_risk`, `high_risk`, `model_type`, `latency_ms`.

| Issue | Fix |
|-------|-----|
| Port in use | Change host port: `-p 8001:8000` |
| Container exits | `docker logs failure-api` |
| 422 | Field names / ranges (e.g. `battery_health_pct` 0–100) |
| Model missing | `model.onnx` + `features.json` next to `main.py` |

**Docker:** same environment everywhere · scale replicas · rollback by image tag.

Step 7 done.
"""
    )
)

# --- Step 8 ---
cells.append(
    md(
        """## Step 8: Measure latency (p50, p95, p99)

Load testing here means **inference serving** (HTTP + ONNX), not training.

**Metrics:** p50 / p95 / p99 (ms), average latency, RPS, error count.

**Concurrency:**
- **Sequential** (1 worker) — one request at a time; clean per-request latency.
- **Parallel** (`--workers` > 1) — many clients at once; shows contention.

**p95** = 95% of successful requests were faster than this value.

### Suggested test matrix

| Scenario | Where | n | Workers |
|----------|-------|---|---------|
| Baseline `/predict` | `latency_test.py` | 500 | 1 |
| Parallel clients | `latency_test.py` | 100 | 10 |
| ONNX vs API | cells below | 500 | 1 |
| Single vs batch | cells below | 100 | 1 |
| 1 vs N Docker replicas | `compare_load.py` | 100 | 1 and 10 |

Use rows from `input/device_telemetry.csv`. For Docker replicas, start containers first (e.g. ports 8002–8006). `compare_load.py` uses **client round-robin** across URLs — that is not the same as a production load balancer.
"""
    )
)

cells.append(
    code(
        """N = 500
sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=N, replace=True)
api_latencies = []
t0_wall = time.perf_counter()
for i in sample_idx:
    row = X_test.iloc[i][FEATURES].to_dict()
    t0 = time.perf_counter()
    requests.post("http://127.0.0.1:8000/predict", json=row, timeout=30).raise_for_status()
    api_latencies.append((time.perf_counter() - t0) * 1000)
api_wall = time.perf_counter() - t0_wall

api_stats = {
    "p50_ms": np.percentile(api_latencies, 50),
    "p95_ms": np.percentile(api_latencies, 95),
    "p99_ms": np.percentile(api_latencies, 99),
    "avg_ms": np.mean(api_latencies),
    "rps": N / api_wall,
}
display(pd.DataFrame([api_stats]).round(3))
"""
    )
)

cells.append(
    code(
        """p50, p95, p99 = api_stats["p50_ms"], api_stats["p95_ms"], api_stats["p99_ms"]
plt.figure(figsize=(7, 4))
plt.hist(api_latencies, bins=40, color="#54A24B", edgecolor="white")
for val, label in [(p50, "p50"), (p95, "p95"), (p99, "p99")]:
    plt.axvline(val, color="black", linestyle="--", label=f"{label}={val:.2f}ms")
plt.xlabel("Latency (ms)")
plt.legend()
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """model_latencies = []
for i in sample_idx:
    row = X_test.iloc[i][FEATURES].values.astype(np.float32).reshape(1, -1)
    t0 = time.perf_counter()
    sess.run(None, {onnx_input_name: row})
    model_latencies.append((time.perf_counter() - t0) * 1000)
display(
    pd.DataFrame(
        {
            "p50_ms": [np.percentile(model_latencies, 50), p50],
            "p95_ms": [np.percentile(model_latencies, 95), p95],
            "avg_ms": [np.mean(model_latencies), np.mean(api_latencies)],
        },
        index=["ONNX only", "API /predict"],
    ).round(3)
)
"""
    )
)

cells.append(
    code(
        """batch100 = X_test.iloc[:100]
t0 = time.perf_counter()
for _, row in batch100.iterrows():
    requests.post(
        "http://127.0.0.1:8000/predict", json=row[FEATURES].to_dict(), timeout=30
    ).raise_for_status()
single_total = time.perf_counter() - t0
t0 = time.perf_counter()
requests.post(
    "http://127.0.0.1:8000/predict/batch",
    json=batch100[FEATURES].to_dict(orient="records"),
    timeout=60,
).raise_for_status()
batch_total = time.perf_counter() - t0
print(f"100 singles: {single_total*1000:.1f} ms | batch(100): {batch_total*1000:.1f} ms")
print(f"Batch speedup: {single_total/batch_total:.2f}x")
"""
    )
)

cells.append(
    md(
        """### CLI load tests (server or Docker must be running)

```bash
# Sequential baseline (adjust --url to your server, e.g. :8002 for Docker)
python lab5/outputs/scripts/latency_test.py \\
  --url http://127.0.0.1:8000 --csv input/device_telemetry.csv -n 500 --warmup 10

# Parallel clients
python lab5/outputs/scripts/latency_test.py \\
  --url http://127.0.0.1:8000 --csv input/device_telemetry.csv -n 100 --workers 10

# 1 container vs N replicas (default replica ports 8002–8006)
python lab5/outputs/scripts/compare_load.py \\
  --csv input/device_telemetry.csv -n 100 --mode both --workers 10
```
"""
    )
)

cells.append(
    code(
        """# Optional: run compare_load if Docker replicas are up (8002–8006 by default)
compare_py = Path(OUTPUT_DIR) / "scripts" / "compare_load.py"
csv_path = PROJECT_ROOT / "input" / "device_telemetry.csv"
if compare_py.is_file() and csv_path.is_file():
    print("Running compare_load.py (skip if no containers on 8002–8006)...")
    result = subprocess.run(
        [
            sys.executable,
            str(compare_py),
            "--csv",
            str(csv_path),
            "-n",
            "100",
            "--mode",
            "both",
            "--workers",
            "10",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print("(compare_load failed — start replica containers or ignore for local-only runs)")
else:
    print("compare_load.py or CSV not found — run Step 6 first")
"""
    )
)

cells.append(
    code(
        """paths = [
    "model/model.onnx", "model/features.json", "model/metrics.json",
    "app/main.py", "app/requirements.txt", "app/Dockerfile", "app/.dockerignore",
    "app/model.onnx", "app/features.json",
    "scripts/call_api.py", "scripts/latency_test.py", "scripts/compare_load.py",
    "lab5_app.zip",
]
print("=== Lab 5 summary ===")
print("Rows:", len(df), f"| Failure rate: {failure_rate:.2f}%")
print("Winner:", winner_name, f"| Test F1: {winner_metrics['f1']:.4f}")
print(f"ONNX: {size_kb:.1f} KB | parity max diff: {max_diff}")
print(f"Latency p50/p95/p99: {p50:.2f} / {p95:.2f} / {p99:.2f} ms")
print("Artifacts:")
for rel in paths:
    p = Path(OUTPUT_DIR) / rel
    if p.exists():
        print(" ", p)
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
print("Stopped uvicorn.")
print("Step 8 done")
"""
    )
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = LAB5_DIR / "lab5_failure_model.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print("Wrote", out, "cells:", len(cells))
