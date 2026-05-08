"""FastAPI application for the EnergyGuard Fairness Audit Service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
TMP_DIR = Path("tmp_uploads")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create required directories on startup."""
    RUNS_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="EnergyGuard Fairness Audit Service",
    description=(
        "Fairness diagnostic microservice for the EnergyGuard TEF "
        "(Horizon Europe GA 101172705). "
        "Accepts a trained sklearn model + tabular CSV + sensitive feature names; "
        "produces a machine-readable fairness report conforming to REPORT_SCHEMA v1.0."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Response models ────────────────────────────────────────────────────────────

class EvaluationCreateResponse(BaseModel):
    """Returned immediately after a job is accepted."""

    job_id: str
    status: str  # "pending"


class JobStatusResponse(BaseModel):
    """Current status of an evaluation job."""

    job_id: str
    status: str  # pending | running | ok | error
    created_at: str
    updated_at: str
    error: str | None = None


class JobConflictResponse(BaseModel):
    """Returned as 409 body so the frontend avoids a second status call."""

    job_id: str
    status: str  # running | pending


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/api/evaluations",
    response_model=EvaluationCreateResponse,
    status_code=202,
    summary="Submit a new fairness evaluation job",
)
async def create_evaluation(
    model_file: UploadFile,
    dataset_file: UploadFile,
    config: UploadFile,
) -> EvaluationCreateResponse:
    """Accept a model, dataset, and YAML config; enqueue a background evaluation.

    Uploaded files are written to a temp directory first. Config path fields are
    overwritten with the resolved temp paths before RunConfig validation. Returns
    the job_id and initial status ("pending").

    Args:
        model_file: Trained sklearn Pipeline (.joblib).
        dataset_file: Tabular dataset (.csv).
        config: YAML RunConfig. model.path and dataset.path are ignored — the
            uploaded file paths are injected automatically.

    Returns:
        EvaluationCreateResponse with job_id and "pending" status.

    Raises:
        HTTPException 422: If the merged RunConfig fails Pydantic validation.
    """
    raise NotImplementedError


@app.get(
    "/api/evaluations/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get evaluation job status",
)
async def get_evaluation_status(job_id: str) -> JobStatusResponse:
    """Return the current status and timestamps for an evaluation job.

    Args:
        job_id: Job identifier returned by POST /api/evaluations.

    Returns:
        JobStatusResponse with current status.

    Raises:
        HTTPException 404: If the job_id is unknown.
    """
    raise NotImplementedError


@app.get(
    "/api/evaluations/{job_id}/metrics",
    summary="Get metrics_user.json payload for a completed evaluation",
)
async def get_evaluation_metrics(job_id: str) -> dict[str, Any]:
    """Return the schema-conforming metrics_user.json for a completed job.

    Args:
        job_id: Job identifier.

    Returns:
        Dict conforming to docs/REPORT_SCHEMA.md.

    Raises:
        HTTPException 404: Job not found.
        HTTPException 409: Job not yet complete; body contains
            {"job_id": ..., "status": "running"|"pending"}.
    """
    raise NotImplementedError


@app.get(
    "/api/evaluations/{job_id}/report",
    response_class=HTMLResponse,
    summary="Get the self-contained HTML report for a completed evaluation",
)
async def get_evaluation_report(job_id: str) -> str:
    """Return the self-contained Plotly+Jinja2 HTML report.

    Args:
        job_id: Job identifier.

    Returns:
        Self-contained HTML string (include_plotlyjs=True).

    Raises:
        HTTPException 404: Job not found.
        HTTPException 409: Job not yet complete; body contains
            {"job_id": ..., "status": "running"|"pending"}.
    """
    raise NotImplementedError
