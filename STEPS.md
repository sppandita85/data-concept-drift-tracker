# data-concept-drift-tracker — Steps 1–15

Day 4 capstone path: laptop failure model, production monitoring, data drift, and demo retrain loop.

**Prerequisites:** Lab 5 model (`model.onnx` + FastAPI + Dockerfile) and training CSV in `input/`.  
**Local defaults:** API http://localhost:8088 · artifacts under `lab7-dryrun/outputs/`.

---

## Lab 7 — Production monitoring & drift (Steps 1–10)

### Step 1 — Log every prediction

Change the Lab 5 API so each `/predict` call writes one line to `logs/predictions.jsonl`: timestamp, device ID, the six inputs, failure risk, high-risk flag, and model version.

**Check:** Call the API a few times; the log file has one line per call (`lab7-dryrun/outputs/logs/predictions.jsonl`).

---

### Step 2 — Add a `/metrics` endpoint

Add request count, error count, and latency using the `prometheus-client` library.

**Check:** http://localhost:8088/metrics or http://localhost:8088/metrics/view shows `failure_*` metrics.

---

### Step 3 — Save the reference data

Copy the training data to `data/reference.csv`. This is what “normal” looks like for drift detection.

**Check:** `lab7-dryrun/outputs/data/reference.csv` exists (created by `python lab7-dryrun/prepare_lab7.py`).

---

### Step 4 — Write the drift job

A small Python job that, every few minutes, reads the last 1,000 predictions from the log, compares each feature with `reference.csv` using Evidently, saves an HTML report, and publishes a drift score per feature for Prometheus (port 9100).

**Check:** `lab7-dryrun/outputs/reports/drift_report.html` and `feature_drift_score` at http://localhost:9100/metrics.

**Code:** `lab7-dryrun/templates/drift_job/drift_check.py`

---

### Step 5 — Start everything with Docker Compose

Run four services together: the API, Prometheus, Grafana, and the drift job.

**Check:** `docker compose ps` (in `lab7-dryrun/outputs/`) shows all four running; Grafana opens at http://localhost:3000 (admin / admin).

| Service      | Role                                      | Host port |
|-------------|-------------------------------------------|-----------|
| `api`       | FastAPI + ONNX, logging, metrics          | 8088      |
| `prometheus`| Scrapes API and drift metrics             | 9090      |
| `grafana`   | Dashboards and alerts                     | 3000      |
| `drift`     | Evidently + drift gauges                  | 9100      |

---

### Step 6 — Normal traffic

Send a few thousand normal laptop readings from the CSV to the API.

**Check:** Drift scores stay low (typically &lt; ~0.1); Evidently shows drift not detected.

```bash
python lab7-dryrun/run_lab7.py traffic
```

---

### Step 7 — Build the Grafana dashboard

Add four panels: p95 latency, error rate, drift score per feature, and the share of high-risk predictions.

**Check:** Dashboard **“Lab 7 — Failure API Monitoring & Drift”** in Grafana.

---

### Step 8 — Add alerts in Grafana

Alert when:

- p95 latency above **200 ms**
- error rate above **2%**
- drift score above **0.25**
- high-risk share above **20%**

**Check:** Alert rules under Grafana → Alerting (Lab 7 folder); firmware demo should fire the drift rule.

---

### Step 9 — Simulate the firmware update

Send traffic again with CPU temperature **8 °C higher** on every laptop.

**Check:** `cpu_temp_avg` drift score rises above **0.25**; high-risk share goes up.

```bash
python lab7-dryrun/run_lab7.py firmware
```

---

### Step 10 — Watch the alert fire

The drift alert turns red on the dashboard. Open the Evidently report to see which feature drifted and by how much.

**Lab 7 deliverables:**

- Grafana screenshot **before** firmware
- Grafana screenshot **after** firmware
- Evidently drift report
- Screenshot of the **fired alert**

---

## Capstone M4 — Closed loop (Steps 11–15)

### Step 11 — Retrain trigger (demo)

When drift is at or above **0.25** (e.g. `cpu_temp_avg`), run a retrain script: train on existing labeled `input/*.csv`, export new ONNX, copy into the Lab 7 API, log a retrain event.

This simulates a retrain job; it does **not** train from unlabeled production log lines alone.

**Check:** Script finishes; API has new model files; entry in `capstone-m4/logs/retrain_events.jsonl`.

```bash
python capstone-m4/check_drift_and_retrain.py
# or: python capstone-m4/retrain_trigger.py
```

---

### Step 12 — Refresh monitoring baseline

After retrain / operational “fix,” update `lab7-dryrun/outputs/data/reference.csv` (copy from `input/` again or an agreed post-fix snapshot).

**Check:** Reference file updated; drift job uses it on the next run.

```bash
python capstone-m4/refresh_baseline.py
```

(Default: `retrain_trigger.py` also refreshes the baseline unless `--no-refresh-baseline` is passed.)

---

### Step 13 — Recovery demo

Stop firmware traffic; run **normal** traffic again so the last 1,000 log lines match the baseline.

**Check:** Evidently drift **down**; Grafana drift panel drops; high-risk share normalizes.

```bash
python lab7-dryrun/run_lab7.py traffic
python lab7-dryrun/run_lab7.py traffic   # optional second batch to fill the 1000-line window
```

---

### Step 14 — Integrated runbook

One script runs the full story: compose up → normal → firmware → alert window → retrain → normal recovery → print URLs.

**Check:** Single rehearsal without manual guesswork; see `capstone-m4/RUNBOOK.md`.

```bash
chmod +x capstone-m4/demo.sh
./capstone-m4/demo.sh
```

Use `FRESH=1 ./capstone-m4/demo.sh` to re-run `prepare_lab7.py` (wipes `lab7-dryrun/outputs/`).

---

### Step 15 — Final presentation

Five to seven slides plus talk track: Lab 5 train/serve → Lab 7 monitor/drift/alert → M4 retrain + baseline + recovery.

**Check:** `capstone-m4/PRESENTATION_OUTLINE.md` and captured artifacts (before / after / alert / retrain / recovery screenshots).

---

## Quick command reference

```bash
python lab7-dryrun/prepare_lab7.py
python lab7-dryrun/run_lab7.py up
python lab7-dryrun/run_lab7.py traffic | firmware | drift-once | down
./capstone-m4/demo.sh
python capstone-m4/check_drift_and_retrain.py   # add --force if drift metrics unavailable
```

## Thresholds (teaching)

| Signal              | Normal   | Alert    |
|---------------------|----------|----------|
| Feature drift score | &lt; ~0.1 | &gt; 0.25 |
| High-risk share     | stable   | &gt; 20%  |
| p95 latency         | low      | &gt; 200 ms |
| Error rate          | ~0       | &gt; 2%    |
