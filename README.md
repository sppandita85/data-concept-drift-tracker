# data-concept-drift-tracker

Laptop **failure-risk** ML pipeline (Lenovo labs): train & serve (Lab 5), optional edge canary (Lab 6), **production monitoring & data drift** (Lab 7), and **Capstone M4** closed-loop demo (retrain + baseline refresh).

## Layout

| Path | Purpose |
|------|---------|
| `input/` | Shared training CSVs |
| `lab5/` | Failure model notebook + training scripts |
| `lab6/` | Edge model + canary router |
| `lab7-dryrun/` | Docker stack: API, Prometheus, Grafana, Evidently drift |
| `capstone-m4/` | Drift-triggered retrain demo + `demo.sh` |

## Quick start

```bash
./setup_local.sh
source .venv/bin/activate

# Lab 5 — train (notebook or scripts in lab5/)
# Lab 7 — monitoring
python lab7-dryrun/prepare_lab7.py
cd lab7-dryrun/outputs && docker compose up -d --build

# Capstone — full story
./capstone-m4/demo.sh
```

## URLs (Lab 7 default ports)

- API: http://localhost:8088/docs  
- Grafana: http://localhost:3000 (admin / admin)  
- Prometheus: http://localhost:9090  
- Drift metrics: http://localhost:9100/metrics  

See **[STEPS.md](STEPS.md)** for the full Step 1–15 checklist, plus [capstone-m4/RUNBOOK.md](capstone-m4/RUNBOOK.md) for the demo script.
