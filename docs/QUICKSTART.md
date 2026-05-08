# EnergyGuard Fairness Audit — Quick Start

## Prerequisites

- Python 3.11+
- `uv` (recommended) or `pip`

---

## Install

```bash
# Clone and install with dev extras
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

---

## Generate Example Data and Baseline Model

These steps are required once before running the smoke test or the CLI example.

```bash
# 1. Generate the synthetic EU energy burden dataset (seed=42, 5000 rows)
python examples/scripts/generate_synthetic_energy_burden.py

# 2. Train the baseline LogisticRegression model
python examples/scripts/train_baseline_model.py
```

Outputs:
- `examples/data/energy_burden_synthetic.csv`
- `examples/models/baseline_logreg.joblib`

---

## CLI Usage

```bash
python -m runner run --config examples/configs/energy_burden.yaml
```

Output is written to `runs/<job_id>/`:
- `config_resolved.yaml` — resolved config
- `run.log`              — structured log
- `metrics.json`         — firehose metrics + provenance
- `metrics_user.json`    — schema-conforming report payload
- `report.html`          — self-contained Plotly report

Override the output directory:

```bash
python -m runner run --config examples/configs/energy_burden.yaml --output /tmp/my-runs
```

---

## API Usage

Start the server:

```bash
uvicorn app.main:app --reload
```

Submit an evaluation:

```bash
curl -X POST http://localhost:8000/api/evaluations \
  -F "model_file=@examples/models/baseline_logreg.joblib" \
  -F "dataset_file=@examples/data/energy_burden_synthetic.csv" \
  -F "config=@examples/configs/energy_burden.yaml"
# → {"job_id": "...", "status": "pending"}
```

Poll status:

```bash
curl http://localhost:8000/api/evaluations/<job_id>/status
```

Fetch metrics:

```bash
curl http://localhost:8000/api/evaluations/<job_id>/metrics | python -m json.tool
```

Fetch HTML report:

```bash
curl http://localhost:8000/api/evaluations/<job_id>/report -o report.html
open report.html
```

---

## Docker (full local stack)

### Start

```bash
docker compose up --build -d
```

Or use the helper script:

```bash
./start_local.sh
```

Services:
| Service | URL |
|---------|-----|
| Fairness Audit API | http://localhost:9006 |
| Swagger UI | http://localhost:9006/docs |
| MLflow UI | http://localhost:9007 |

### Health check

```bash
curl http://localhost:9006/health
# → {"status":"ok"}
```

### Run a test audit against the container

```bash
# 1. Submit evaluation
JOB=$(curl -s -X POST http://localhost:9006/api/evaluations \
  -F "model_file=@examples/models/baseline_logreg.joblib" \
  -F "dataset_file=@examples/data/energy_burden_synthetic.csv" \
  -F "config=@examples/configs/energy_burden.yaml" | python -m json.tool)
echo "$JOB"
JOB_ID=$(echo "$JOB" | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll until complete
until [ "$(curl -s http://localhost:9006/api/evaluations/$JOB_ID/status | python -c "import sys,json; print(json.load(sys.stdin)['status'])")" = "ok" ]; do
  echo "waiting..."; sleep 2
done

# 3. Fetch metrics
curl -s http://localhost:9006/api/evaluations/$JOB_ID/metrics | python -m json.tool

# 4. Fetch HTML report
curl -s http://localhost:9006/api/evaluations/$JOB_ID/report -o report.html
```

### Stop

```bash
docker compose down
```

---

## Run Tests

```bash
pytest                          # all tests
pytest tests/test_config.py    # config model tests
pytest tests/test_metrics.py   # threshold tests
pytest tests/test_smoke.py     # end-to-end smoke (requires example data)
```

---

## MLFLOW_TRACKING_URI

The service reads `MLFLOW_TRACKING_URI` from the environment. No hardcoded URIs.

```bash
# Local docker-compose MLflow
export MLFLOW_TRACKING_URI=http://localhost:5001

# Production (set at integration time)
export MLFLOW_TRACKING_URI=https://mlflow.toolbox.epu.ntua.gr
```

MLflow logging is **not active in v1**. The env var is read but no calls are made.
Active logging is wired in v1.1.
