# Capstone M4 — presentation outline (5–7 slides)

## Slide 1 — Problem

- Lenovo fleet: predict **failure in 30 days** from device telemetry.
- Models decay when **reality changes** (firmware, seasonality, hardware mix).
- Need **monitoring + drift detection + a response playbook**.

## Slide 2 — Lab 5: train & serve

- Labeled CSV in `input/` → Random Forest / XGBoost → **ONNX**.
- **FastAPI** + Docker: `/predict` returns risk score.

## Slide 3 — Lab 7: production monitoring

- Log every prediction → **`predictions.jsonl`**.
- **Prometheus** metrics: latency, errors, prediction volume.
- **`reference.csv`** = training snapshot = “normal.”
- **Evidently** drift job vs last 1,000 predictions.

## Slide 4 — Dashboard & alerts

- **Grafana** (4 panels): p95 latency, error rate, drift by feature, high-risk share.
- Alerts: latency &gt; 200 ms, errors &gt; 2%, **drift &gt; 0.25**, high-risk &gt; 20%.

## Slide 5 — Incident demo (firmware)

- Script sends **CPU +8°C** on every device.
- **`cpu_temp_avg` drift ~0.98**; high-risk share rises; alert fires.
- Evidently report: which feature drifted.

## Slide 6 — Capstone M4: respond & recover

- **Step 11:** Drift gate → **retrain script** on labeled `input/` (demo pipeline; real life = new labels).
- **Step 12:** Refresh **`reference.csv`** baseline.
- **Step 13:** Normal traffic → drift **down** on Grafana/Evidently.
- Show **`retrain_events.jsonl`** + “Last retrain” panel.

## Slide 7 — Lessons

- Monitor **inputs**, not only model accuracy.
- **Fixed baseline** until you consciously refresh it after a fix.
- Drift alerts should tie to a **runbook** (retrain, rollback, baseline update).
- Production retrain needs **labeled data** — logs alone are not enough.

## Artifacts to show live

1. http://localhost:3000 — dashboard time range covering before / spike / recovery  
2. `lab7-dryrun/outputs/reports/drift_report.html`  
3. `capstone-m4/logs/retrain_events.jsonl` (last line)
