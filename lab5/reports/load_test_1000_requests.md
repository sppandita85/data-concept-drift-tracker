# Load test results — 1000 requests

**Date:** 2026-10-29  
**Machine:** MacBook (local Docker Desktop)  
**Image:** `laptop-failure-api:v1`  
**Payloads:** Random rows from `input/device_telemetry.csv` (6 feature fields)  
**Endpoint:** `POST /predict`  
**Errors:** 0 on all runs below  

Also running locally (not used in these numbers): uvicorn on ports 8000 and 8001.

---

## 1. Single Docker replica (`latency_test.py`)

**Target:** `http://127.0.0.1:8002` (`failure-api-1`)  
**Warmup:** 20 requests (not counted)

| Mode | Workers | OK | p50 (ms) | p95 (ms) | p99 (ms) | avg (ms) | RPS |
|------|---------|-----|----------|----------|----------|----------|-----|
| Sequential | 1 | 1000 | 2.67 | 7.73 | 12.37 | 3.38 | 291.20 |
| Parallel | 10 | 1000 | 14.05 | 25.62 | 30.36 | 14.74 | 654.11 |

```bash
python lab5/outputs/scripts/latency_test.py \
  --url http://127.0.0.1:8002 --csv input/device_telemetry.csv \
  -n 1000 --warmup 20 --workers 1

python lab5/outputs/scripts/latency_test.py \
  --url http://127.0.0.1:8002 --csv input/device_telemetry.csv \
  -n 1000 --warmup 20 --workers 10
```

---

## 2. One container vs five replicas (`compare_load.py`)

**Single target:** `http://127.0.0.1:8002`  
**Replicas (client round-robin):** ports 8002–8006 (`failure-api-1` … `failure-api-5`)  

| Scenario | requests | ok | errors | total (s) | RPS | avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|----------|----------|-----|--------|-----------|-----|----------|----------|----------|----------|----------|
| Sequential → 1 container | 1000 | 1000 | 0 | 2.61 | 382.43 | 2.57 | 2.50 | 3.28 | 4.19 | 15.38 |
| Sequential → 5 containers (RR) | 1000 | 1000 | 0 | 2.15 | 465.35 | 2.11 | 1.96 | 3.31 | 4.17 | 5.74 |
| Parallel (10 workers) → 1 container | 1000 | 1000 | 0 | 1.38 | 722.41 | 13.27 | 12.37 | 22.43 | 31.29 | 52.91 |
| Parallel (10 workers) → 5 containers (RR) | 1000 | 1000 | 0 | 1.11 | 897.73 | 10.64 | 10.05 | 17.40 | 21.57 | 28.93 |

**Sequential wall-clock speedup (5 replicas vs 1):** ~1.22×  
**Parallel throughput speedup (5 vs 1):** ~1.24× (722 → 898 RPS)

```bash
python lab5/outputs/scripts/compare_load.py \
  --csv input/device_telemetry.csv -n 1000 --mode both --workers 10
```

---

## Notes for class

- **Sequential** = one request at a time; best for per-request latency (p50 ~2–3 ms on one container).
- **Parallel (10 workers)** = concurrent clients; higher RPS, higher latency percentiles.
- **Round-robin across ports** is a teaching stand-in for load balancing, not production routing (no single URL / nginx / K8s Service).
- **p95** = 95% of successful requests were faster than this value.

---

## Re-run checklist

1. Containers up: `docker ps --filter name=failure-api`
2. Health: `curl http://127.0.0.1:8002/health`
3. From repo root with venv: `source .venv/bin/activate`
4. Run commands above; update this file with new numbers if you change hardware or replica count.
