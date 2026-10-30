# Capstone M4 demo runbook (~15 minutes)

## Before you start

1. Python venv with repo [requirements.txt](../requirements.txt) (includes Evidently for local drift-once).
2. Lab 5 model already trained **or** retrain will create `lab5/outputs/model/`.
3. Docker Desktop running.

## One-command rehearsal

```bash
# From repo root (does not wipe outputs unless FRESH=1)
chmod +x capstone-m4/demo.sh
./capstone-m4/demo.sh
```

Fresh staging (deletes `lab7-dryrun/outputs/`):

```bash
FRESH=1 ./capstone-m4/demo.sh
```

## Manual steps (same story)

```bash
python lab7-dryrun/prepare_lab7.py          # if outputs missing
cd lab7-dryrun/outputs && docker compose up -d --build

python lab7-dryrun/run_lab7.py traffic      # Step 6 — screenshot Grafana (calm)
python lab7-dryrun/run_lab7.py firmware     # Step 9 — wait ~90s — screenshot + alert
python capstone-m4/check_drift_and_retrain.py   # Steps 11–12 (or --force)
python lab7-dryrun/run_lab7.py traffic      # Step 13 — twice to fill 1000-line window
python lab7-dryrun/run_lab7.py drift-once   # Evidently + local report
```

## What to say (talk track)

1. **Lab 5** — We trained on labeled fleet CSV and serve ONNX via FastAPI.
2. **Lab 7 Step 1–2** — Every prediction is logged; Prometheus scrapes SLO metrics.
3. **Step 3** — `reference.csv` is the frozen “normal” baseline for drift.
4. **Step 4–5** — Evidently job + Grafana on Docker Compose.
5. **Step 9** — Firmware sim adds 8°C; **`cpu_temp_avg` drift** spikes; alert fires.
6. **Step 11** — Alert triggers **retrain playbook** (labeled `input/` — demo stand-in for a new dataset).
7. **Step 12** — Refresh baseline file; **Step 13** — normal traffic again; drift **drops** → closed loop.

## Screenshots checklist

- [ ] Grafana **before** firmware (drift low)
- [ ] Grafana **after** firmware (drift + high-risk up)
- [ ] Grafana **alert firing** (drift rule)
- [ ] Evidently HTML (**Detected** on `cpu_temp_avg`)
- [ ] Grafana **after recovery** (drift down)
- [ ] Line in `capstone-m4/logs/retrain_events.jsonl`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8088 busy | `docker compose down` in old lab7 folder; or set `LAB7_API_PORT` |
| Drift never drops after recovery | Send more normal traffic (2×3000) or wait for drift interval (60s) |
| Retrain but API old version | Retrain runs `docker compose up -d --build api` — check build logs |
| `check_drift` can’t read score | Ensure drift container up; or use `--force` |
