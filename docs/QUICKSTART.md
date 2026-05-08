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

```bash
./start_local.sh
```

Services:
| Service | URL |
|---------|-----|
| Fairness Audit API | http://localhost:8080 |
| Swagger UI | http://localhost:8080/docs |
| MLflow UI | http://localhost:5001 |

Stop: `docker compose down`

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
