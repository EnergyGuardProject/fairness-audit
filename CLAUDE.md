# EnergyGuard Fairness Audit Service

## Project Context
Fairness-audit microservice for the EnergyGuard project (Horizon Europe 
GA 101172705), a TEF for trustworthy AI in the energy sector. Component 
of T5.2 (Trustworthy AI Acceptance Environment) under WP5.

Sibling service to the Adversarial Robustness Service (Alex Tzortzis). 
Both follow the same architectural pattern; reports must use a unified 
schema so the EnergyGuard frontend can render either with the same 
component tree.

## What It Does
Input: trained ML model + dataset + sensitive/grouping attributes
Process: compute fairness or subgroup-performance metrics via Fairlearn
Output: machine-readable metrics JSON + HTML report

DIAGNOSTIC ONLY. No mitigation in v1. No model retraining.

## Dual Track
Submissions are classified into one of two tracks before evaluation:
- **Fairness track**: consumer-facing models with sensitive attributes 
  (gender, age, income proxies). Computes classical fairness metrics 
  (DP, EO, EOdds) + AI Act severity scoring.
- **Subgroup-performance track**: operational/asset-level models. 
  Computes performance parity across operational subgroups (asset 
  class, region, season). No fairness vocabulary in the report.

Both tracks use Fairlearn's MetricFrame underneath.

## Stack
- Python 3.11+
- FastAPI (HTTP) + Typer (CLI)
- Fairlearn (only fairness library — no AIF360, no Aequitas)
- scikit-learn for model handling
- Pandas for data
- Plotly + Jinja2 for HTML reports
- Pydantic v2 for config and schemas
- pytest for tests
- uv or pip-tools for deps
- Docker for packaging

## Architectural Reference
Mirror the structure of model-robustness-service (Alex Tzortzis's 
analogue for adversarial robustness). Specifically reuse:
- YAML config + Pydantic RunConfig validation
- `runs/<job_id>/` filesystem job pattern
- Two-tier metrics: full `metrics.json` + projected `metrics_user.json`
- Standardised report payload schema:
    {report_meta, *_setup, *_summary, charts, warnings}
- ModelAdapter abstraction over sklearn / PyTorch / MLflow loading

DO NOT copy the robustness service's code verbatim — it has known 
issues (no tests, hardcoded paths, leading-underscore fields mixed 
with public ones in TypedDicts, over-engineered path resolution). 
Use it as a structural template only.

## Conventions
- Type hints everywhere; mypy-clean
- Pydantic models for all API request/response bodies
- Snake_case for Python, kebab-case for CLI commands
- Docstrings with Args / Returns / Raises on all public functions
- Logging via the `logging` module — no print statements
- Tests required for non-trivial logic; smoke test must always pass

## Constraints
- Do NOT add mitigation algorithms in v1
- Do NOT add model retraining
- Do NOT add authentication in v1
- Use Fairlearn as the single fairness library
- Do NOT hardcode paths under /home/<user>/
- Reports must work for classification, regression, and 
  subgroup-performance tracks
- Report payload schema must match robustness service shape

## Current Focus (v1)
1. Pydantic config models (RunConfig, FairnessConfig, ModelConfig, ...)
2. Dataset and model loaders (sklearn + MLflow first, PyTorch later)
3. Use-case classifier (sensitive vs operational attributes)
4. Fairness engine (MetricFrame wrapper, classification + regression)
5. Subgroup-performance engine (operational track)
6. AI Act severity scorer
7. Us
