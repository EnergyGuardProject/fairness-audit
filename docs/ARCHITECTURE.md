# EnergyGuard Fairness Audit Service — Architecture

## Overview

Single-service microservice that accepts a trained sklearn model + tabular CSV dataset
+ a list of sensitive feature column names, computes Fairlearn-based fairness metrics,
and produces a machine-readable report conforming to `docs/REPORT_SCHEMA.md`.

Two entry points share the same pipeline:

```
CLI  →  runner/cli.py  ┐
                        ├─→  runner/pipeline.py  →  engine.py  →  metrics/  →  reporting/
API  →  app/main.py   ┘
```

---

## Job Pattern — `runs/<job_id>/`

Every evaluation creates an isolated output directory:

```
runs/
└── 20260507_143012_a1b2c3d4/
    ├── config_resolved.yaml   # RunConfig with absolute paths, written before compute
    ├── run.log                # Structured log for this job
    ├── metrics.json           # Firehose — all numbers + provenance block
    ├── metrics_user.json      # Schema-conforming payload (docs/REPORT_SCHEMA.md)
    └── report.html            # Self-contained Plotly+Jinja2 report
```

`job_id` format: `YYYYMMDD_HHMMSS_<8-char hex uuid>`.

The API persists job state in `status.json` inside the job directory. This file is
readable by the CLI path as well, so both entry points share the same status format.

---

## TaskStrategy Pattern

```python
# runner/task_strategy.py
class TaskStrategy(ABC):
    def execute(config: RunConfig, job_dir: Path) -> None: ...

class ClassificationFairnessStrategy(TaskStrategy): ...
# RegressionFairnessStrategy — v1.1
```

`pipeline.py` selects the strategy based on the task type inferred from the dataset
(v1: always `ClassificationFairnessStrategy`). This mirrors the robustness service
pattern and keeps the pipeline open/closed for new task types.

---

## ModelAdapter

`runner/loaders/model_adapter.py` wraps any loaded model object behind a uniform
`predict(X: pd.DataFrame) -> np.ndarray` interface. In v1, only `sklearn` backend
(`.joblib` files) is supported. The adapter performs duck-typing validation
(`hasattr(obj, "predict")`); it logs an info-level note if the loaded object is not
a `sklearn.pipeline.Pipeline` but does not fail.

---

## API File-Upload Flow

```
POST /api/evaluations  (multipart/form-data)
  1. Save model_file  →  tmp_uploads/<job_id>/model.joblib
  2. Save dataset_file →  tmp_uploads/<job_id>/dataset.csv
  3. Parse config YAML bytes
  4. Inject resolved paths into config dict (overrides any paths in the YAML)
  5. RunConfig.model_validate(config_dict)  — raises HTTP 422 on invalid config
  6. Persist config_resolved.yaml + initial state
  7. Enqueue BackgroundTask → pipeline.run(config, job_dir)
  8. Return 202 { job_id, status: "pending" }
```

409 responses are raised via FastAPI's `HTTPException`. The body FastAPI produces is:

```json
{ "detail": { "job_id": "<job_id>", "status": "pending" | "running" | "error" } }
```

Clients must read `response.json()["detail"]` (not the top-level body) to get the
job state. This lets the frontend skip a redundant GET /status call.

---

## report_meta.status — "partial" reservation

The shared schema (`docs/REPORT_SCHEMA.md`) defines `report_meta.status` as
`"ok" | "partial" | "error"`. The fairness-audit service emits only `"ok"` or
`"error"` in v1 — evaluation is all-or-nothing. `"partial"` is reserved for future
services where a timeout or resource limit may cause incomplete computation (e.g., a
robustness service that completes only some attack types before timing out).

This asymmetry is intentional: the schema is shared across the T5.2 service family;
individual services declare which status values they emit in their own docs.

---

## MLflow Integration

v1: local MLflow container exists in `docker-compose.yml` for shared infrastructure.
The fairness service does **not** log to MLflow in v1. Wired in v1.1.

Tracking URI is configured exclusively via the `MLFLOW_TRACKING_URI` environment
variable. No hardcoded URIs anywhere in the codebase.

Production target: `https://mlflow.toolbox.epu.ntua.gr` — switch at integration time
by setting the env var; no code change required.

---

## PYTHONPATH / Package Layout

The `fairness-backend/` directory holds the `runner` and `app` Python packages.

- **Installed**: `pip install .` via `pyproject.toml` makes `runner` and `app`
  importable from site-packages (hatchling resolves the `fairness-backend/` prefix).
- **Development**: `pytest` uses `pythonpath = ["fairness-backend"]` so tests import
  the packages directly from source without installing.
- **Docker**: packages are installed in the build stage; the runtime stage copies
  site-packages. No `PYTHONPATH` manipulation needed in Docker.
