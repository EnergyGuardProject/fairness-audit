# EnergyGuard Fairness Audit Service

Fairness diagnostic microservice for the EnergyGuard project
([Horizon Europe GA 101172705](https://cordis.europa.eu/project/id/101172705)).
Part of **T5.2 — Trustworthy AI Acceptance Environment** under WP5.

Accepts a trained scikit-learn classification model, a CSV dataset, and a list of
sensitive feature column names. Computes group fairness metrics and produces a
machine-readable JSON report plus a self-contained HTML report. **Diagnostic only —
no mitigation or retraining in v1.**

> ⚠️ **Trust model:** v1 has no authentication. The service deserialises uploaded
> `.joblib` models with `joblib.load` (= `pickle` = arbitrary code execution).
> Run only on a trusted network behind an authenticating gateway.
> See [`SECURITY.md`](SECURITY.md) for details.

---

## What It Does

```
model (.joblib) + dataset (.csv) + config (.yaml)
        │
        ▼
  FairnessEngine
  (NumPy, group-level)
        │
        ├── metrics.json          ← full firehose + provenance
        ├── metrics_user.json     ← schema-conforming report payload
        └── report.html           ← self-contained Plotly report
```

**Primary metrics** (pass / warn / fail):

| Metric | What it measures |
|--------|-----------------|
| Demographic Parity Difference | Gap in positive prediction rates between groups |
| Equalized Odds Difference | Largest FPR or FNR gap across groups (binary only) |
| Selection Rate Ratio | Ratio of lowest to highest selection rate across groups |

**Diagnostic metrics** (per-group, informational):
accuracy, selection rate, false positive rate, false negative rate.

---

## Quick Start

### Docker (recommended)

```bash
# Build and start the full stack (API on :9006, MLflow UI on :9007)
docker compose up --build -d

# Health check
curl http://localhost:9006/health
# → {"status":"ok"}

# Submit an evaluation
JOB_ID=$(curl -s -X POST http://localhost:9006/api/evaluations \
  -F "model_file=@examples/models/baseline_logreg.joblib" \
  -F "dataset_file=@examples/data/energy_burden_synthetic.csv" \
  -F "config=@examples/configs/energy_burden.yaml" \
  | python -m json.tool | python -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Poll until complete
until [ "$(curl -s http://localhost:9006/api/evaluations/$JOB_ID/status \
  | python -c "import sys,json; print(json.load(sys.stdin)['status'])")" = "ok" ]; do
  echo "waiting..."; sleep 2
done

# Fetch the report
curl -s http://localhost:9006/api/evaluations/$JOB_ID/metrics | python -m json.tool
curl -s http://localhost:9006/api/evaluations/$JOB_ID/report -o report.html
```

### CLI

```bash
# Install
pip install -e ".[dev]"

# Generate example data and model (once)
python examples/scripts/generate_synthetic_energy_burden.py
python examples/scripts/train_baseline_model.py

# Run audit
python -m runner run --config examples/configs/energy_burden.yaml
# → runs/<job_id>/{config_resolved.yaml, run.log, metrics.json, metrics_user.json, report.html}
```

---

## Configuration

```yaml
run_name: "My audit run"
description: "Optional description"

dataset:
  path: path/to/dataset.csv
  format: csv
  target_column: is_burdened        # binary label column

model:
  path: path/to/model.joblib
  backend: sklearn

fairness:
  sensitive_features:
    - income_decile
    - country
    - age_head_band
  fairness_metrics:                  # defaults shown
    - demographic_parity_difference
    - equalized_odds_difference
    - selection_rate_ratio
  base_metric: accuracy
  intersectional: false              # true → cross-product groups
  parity_thresholds:                 # override fail boundaries (optional)
    demographic_parity_difference: 0.15
```

See [`docs/FAIRNESS_METRICS.md`](docs/FAIRNESS_METRICS.md) for threshold definitions
and the headline scoring rubric.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/api/evaluations` | Submit model + dataset + config; returns `job_id` |
| `GET`  | `/api/evaluations/{job_id}/status` | Poll job status (`pending` → `running` → `ok`/`error`) |
| `GET`  | `/api/evaluations/{job_id}/metrics` | Fetch `metrics_user.json` payload (409 if not complete) |
| `GET`  | `/api/evaluations/{job_id}/report` | Fetch self-contained HTML report (409 if not complete) |

Interactive docs: `http://localhost:9006/docs`

The report payload conforms to [`docs/REPORT_SCHEMA.md`](docs/REPORT_SCHEMA.md) — the
same schema used by the sibling Adversarial Robustness Service, so the EnergyGuard
frontend renders both with one component tree.

---

## Stack

| Layer | Library |
|-------|---------|
| HTTP service | FastAPI + Uvicorn |
| CLI | Typer |
| Fairness metrics | NumPy (group-level); Fairlearn wired in v1.1 |
| Model loading | scikit-learn + joblib |
| Data | Pandas |
| Reports | Plotly + Jinja2 |
| Config / schemas | Pydantic v2 |
| Packaging | Hatchling (`pyproject.toml`) |
| Container | Docker multi-stage build |

---

## Development

```bash
pip install -e ".[dev]"

# All tests
pytest

# Specific suites
pytest tests/test_config.py     # Pydantic config validation
pytest tests/test_engine.py     # FairnessEngine unit tests
pytest tests/test_metrics.py    # threshold / scoring tests
pytest tests/test_api.py        # FastAPI endpoint tests
pytest tests/test_smoke.py      # end-to-end smoke (requires example data)
```

---

## Repo Layout

```
fairness-audit/
├── Dockerfile                     # Multi-stage build (context: repo root)
├── docker-compose.yml             # API on :9006, MLflow on :9007
├── pyproject.toml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FAIRNESS_METRICS.md        # metric definitions and thresholds
│   ├── QUICKSTART.md
│   └── REPORT_SCHEMA.md           # shared output schema (v1.0)
├── examples/
│   ├── configs/energy_burden.yaml
│   ├── data/energy_burden_synthetic.csv
│   ├── models/baseline_logreg.joblib
│   └── scripts/
│       ├── generate_synthetic_energy_burden.py
│       └── train_baseline_model.py
├── fairness-backend/
│   ├── app/main.py                # FastAPI application
│   └── runner/
│       ├── cli.py                 # Typer CLI
│       ├── pipeline.py            # orchestration
│       ├── engine.py              # FairnessEngine (core compute)
│       ├── config/models.py       # Pydantic RunConfig
│       ├── loaders/               # model + dataset loaders
│       ├── metrics/               # thresholds, firehose, user projection
│       └── reporting/             # Plotly + Jinja2 HTML renderer
└── tests/
```

---

## v1 Scope

- [x] CSV dataset + sklearn model loading
- [x] Group fairness metrics (DPD, EOD, SRR) with configurable thresholds
- [x] Intersectional mode
- [x] Two-tier output: `metrics.json` (firehose) + `metrics_user.json` (schema-conforming)
- [x] Self-contained HTML report (Plotly)
- [x] CLI and FastAPI service with background-task job pattern
- [x] Docker multi-stage build + docker-compose stack

**Out of scope for v1:** MLflow logging (v1.1), regression fairness (v1.1),
tertiary heatmap chart (v1.1), PyTorch model loader (v2), mitigation, authentication, PDF export.

---

## Acknowledgements

Funded by the European Union — Horizon Europe Grant Agreement No. 101172705 (EnergyGuard).
Views and opinions expressed are those of the author(s) only and do not necessarily
reflect those of the European Union or the European Research Executive Agency (REA).
