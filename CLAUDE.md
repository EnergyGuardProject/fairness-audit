# EnergyGuard Fairness Audit Service

## Project Context
Fairness-audit microservice for the EnergyGuard project (Horizon Europe 
GA 101172705), a TEF for trustworthy AI in the energy sector. Component 
of T5.2 (Trustworthy AI Acceptance Environment) under WP5.

Sibling service to the Adversarial Robustness Service (Alex Tzortzis). 
Both follow the same architectural pattern. Reports use a unified schema 
(see docs/REPORT_SCHEMA.md) so the EnergyGuard frontend can render either 
service with one component tree.

## What It Does
Input: trained sklearn classification model + CSV dataset + sensitive 
feature column names.
Process: compute fairness metrics via Fairlearn (MetricFrame).
Output: machine-readable metrics JSON + HTML report.

DIAGNOSTIC ONLY. No mitigation in v1. No model retraining.

## Stack
- Python 3.11+
- FastAPI (HTTP) + Typer (CLI)
- Fairlearn (only fairness library — no AIF360, no Aequitas)
- scikit-learn for model handling
- Pandas for data
- Plotly + Jinja2 for HTML reports
- Pydantic v2 for config and schemas
- pytest for tests
- Docker for packaging

## Architectural Reference
Mirror the structure of model-robustness-service (Alex Tzortzis's 
analogue for adversarial robustness). Specifically reuse:
- YAML config + Pydantic RunConfig validation
- runs/<job_id>/ filesystem job pattern
- Two-tier metrics: full metrics.json + projected metrics_user.json
- Standardised report payload schema (see docs/REPORT_SCHEMA.md)
- ModelAdapter abstraction over model loaders
- FastAPI service shape with background-task job pattern

DO NOT copy the robustness service's code verbatim. Use it as a 
structural template only.

## v1 Scope (Current Focus)
1. Pydantic config models (RunConfig, DatasetConfig, ModelConfig, 
   FairnessConfig)
2. Sklearn model loader + CSV dataset loader
3. Fairness engine for classification (Fairlearn MetricFrame wrapper)
4. Two-tier metrics output (metrics.json + metrics_user.json)
5. metrics_user.json MUST conform to docs/REPORT_SCHEMA.md
6. HTML report renderer (Jinja2 + Plotly, self-contained file)
7. CLI: `python -m runner run --config <path>`
8. FastAPI service: POST /api/evaluations, GET /api/evaluations/{job_id}/status,
   GET /api/evaluations/{job_id}/metrics, GET /api/evaluations/{job_id}/report
9. Dockerfile + docker-compose with local MLflow container
10. Smoke test using the Adult dataset

## Out of Scope for v1
- MLflow loader (defer to v1.1; local MLflow container exists for future use)
- PyTorch loader (defer to v2)
- Regression fairness (defer to v1.1)
- Subgroup-performance track (defer to v1.2)
- AI Act compliance scoring (explicitly excluded)
- UI / frontend (separate repo)
- Authentication
- Mitigation
- PDF export

## Conventions
- Type hints everywhere; mypy-clean
- Pydantic v2 models for all API request/response bodies and config
- Snake_case for Python, kebab-case for CLI commands
- Docstrings with Args / Returns / Raises on all public functions
- Logging via the `logging` module — no print statements
- Tests required for non-trivial logic; smoke test must always pass
- No hardcoded paths under /home/<user>/

## MLflow Integration
- v1: local MLflow container via docker-compose (mirror robustness service pattern)
- Tracking URI configured ONLY via `MLFLOW_TRACKING_URI` env var
- Production target: https://mlflow.toolbox.epu.ntua.gr (switch at integration time)
- Code must never hardcode a tracking URI or assume local vs remote
- v1 service does NOT log to MLflow yet — wired in v1.1

## Repo Layout (Target)
fairness-audit/
├── CLAUDE.md
├── README.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── start_local.sh
├── docs/
│   ├── ARCHITECTURE.md
│   ├── REPORT_SCHEMA.md
│   ├── FAIRNESS_METRICS.md
│   └── QUICKSTART.md
├── examples/
│   ├── configs/
│   │   └── energy_burden.yaml
│   ├── data/
│   │   └── energy_burden_synthetic.csv
│   ├── models/
│   │   └── baseline_logreg.joblib
│   └── scripts/
│       ├── generate_synthetic_energy_burden.py
│       └── train_baseline_model.py
├── fairness-backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── runner/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── engine.py
│       ├── task_strategy.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── models.py
│       ├── loaders/
│       │   ├── __init__.py
│       │   ├── dataset_loaders.py
│       │   ├── model_loaders.py
│       │   ├── model_adapter.py
│       │   └── input_schema.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── compute.py
│       │   ├── classification.py
│       │   ├── thresholds.py
│       │   └── user_metrics.py
│       └── reporting/
│           ├── __init__.py
│           ├── html_report.py
│           └── templates/
│               └── report.html.j2
└── tests/
    ├── fixtures/
    │   └── report_schema_validator.py
    ├── test_config.py
    ├── test_engine.py
    ├── test_metrics.py
    └── test_smoke.py
