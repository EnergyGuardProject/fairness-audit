"""FastAPI application for the EnergyGuard Fairness Audit Service."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from runner.config.models import RunConfig
from runner.pipeline import run_evaluation
from runner.utils import generate_job_id, write_status

logger = logging.getLogger(__name__)

RUNS_DIR = Path("runs")
TMP_DIR = Path("tmp_uploads")

# Reject uploads larger than 500 MB to prevent OOM on the container.
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create required directories on startup."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
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


def _read_status(job_id: str) -> dict[str, Any]:
    """Return parsed status.json for job_id; raise HTTP 404 if not found.

    Args:
        job_id: Job identifier.

    Returns:
        Parsed status dict.

    Raises:
        HTTPException 404: If the job directory or status file does not exist.
    """
    status_path = RUNS_DIR / job_id / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return json.loads(status_path.read_text(encoding="utf-8"))


# ── Response models ────────────────────────────────────────────────────────────

class EvaluationCreateResponse(BaseModel):
    """Returned immediately after a job is accepted (HTTP 202)."""

    job_id: str
    status: Literal["pending"]


class JobStatusResponse(BaseModel):
    """Current status of an evaluation job."""

    job_id: str
    status: Literal["pending", "running", "ok", "error"]
    created_at: str
    updated_at: str
    error: str | None = None


class JobConflictResponse(BaseModel):
    """409 detail body — lets the frontend skip a redundant status call."""

    job_id: str
    status: Literal["pending", "running", "error"]


# ── Background task ────────────────────────────────────────────────────────────

def _evaluation_task(
    config: RunConfig,
    job_dir: Path,
    job_id: str,
    tmp_model: Path,
    tmp_dataset: Path,
) -> None:
    """Run the evaluation pipeline, then clean up uploaded temp files.

    Args:
        config: Validated RunConfig with injected temp-file paths.
        job_dir: Job output directory under RUNS_DIR.
        job_id: Job identifier string.
        tmp_model: Path to the temp model file (deleted after evaluation).
        tmp_dataset: Path to the temp dataset file (deleted after evaluation).
    """
    try:
        run_evaluation(config, job_dir, job_id)
    finally:
        for p in (tmp_model, tmp_dataset):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            tmp_model.parent.rmdir()
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}


@app.post(
    "/api/evaluations",
    response_model=EvaluationCreateResponse,
    status_code=202,
    summary="Submit a new fairness evaluation job",
)
async def create_evaluation(
    background_tasks: BackgroundTasks,
    model_file: UploadFile,
    dataset_file: UploadFile,
    config: UploadFile,
) -> EvaluationCreateResponse:
    """Accept a model, dataset, and YAML config; enqueue a background evaluation.

    Uploaded files are written to TMP_DIR/<job_id>/. The config's model.path
    and dataset.path are overridden with the resolved temp paths before
    RunConfig validation.

    Args:
        background_tasks: FastAPI BackgroundTasks injected by the framework.
        model_file: Trained sklearn Pipeline (.joblib). Max 500 MB.
        dataset_file: Tabular dataset (.csv). Max 500 MB.
        config: YAML RunConfig. model.path and dataset.path are ignored —
            the uploaded file paths are injected automatically.

    Returns:
        EvaluationCreateResponse with job_id and "pending" status.

    Raises:
        HTTPException 400: If the YAML config cannot be parsed.
        HTTPException 413: If any uploaded file exceeds 500 MB.
        HTTPException 422: If the merged RunConfig fails Pydantic validation.
    """
    job_id = generate_job_id()

    # Read and validate upload sizes BEFORE creating any directories so that
    # a 413 rejection never leaves orphaned empty dirs on disk.
    model_bytes = await model_file.read()
    if len(model_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="model_file exceeds 500 MB limit")

    dataset_bytes = await dataset_file.read()
    if len(dataset_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="dataset_file exceeds 500 MB limit")

    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    tmp_job_dir = TMP_DIR / job_id
    tmp_job_dir.mkdir(parents=True, exist_ok=True)

    tmp_model_path = tmp_job_dir / (model_file.filename or "model.joblib")
    tmp_dataset_path = tmp_job_dir / (dataset_file.filename or "dataset.csv")
    tmp_model_path.write_bytes(model_bytes)
    tmp_dataset_path.write_bytes(dataset_bytes)

    config_bytes = await config.read()
    try:
        raw = yaml.safe_load(config_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Config YAML must be a mapping")

    raw.setdefault("dataset", {})["path"] = str(tmp_dataset_path)
    raw.setdefault("model", {})["path"] = str(tmp_model_path)

    try:
        run_config = RunConfig.model_validate(raw)
    except ValidationError as exc:
        # exc.errors() may contain non-serializable ctx.error (ValueError) objects.
        # Round-tripping through exc.json() guarantees a serializable structure.
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_status(job_dir, {
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
    })

    background_tasks.add_task(
        _evaluation_task, run_config, job_dir, job_id, tmp_model_path, tmp_dataset_path,
    )

    return EvaluationCreateResponse(job_id=job_id, status="pending")


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
    data = _read_status(job_id)
    updated_at = (
        data.get("completed_at")
        or data.get("updated_at")
        or data["created_at"]
    )
    return JobStatusResponse(
        job_id=job_id,
        status=data["status"],
        created_at=data["created_at"],
        updated_at=updated_at,
        error=data.get("error"),
    )


@app.get(
    "/api/evaluations/{job_id}/metrics",
    responses={409: {"model": JobConflictResponse}},
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
        HTTPException 409: Job not yet complete; detail has shape
            {"job_id": ..., "status": "pending"|"running"|"error"}.
    """
    data = _read_status(job_id)
    if data["status"] != "ok":
        raise HTTPException(
            status_code=409,
            detail={"job_id": job_id, "status": data["status"]},
        )
    metrics_path = RUNS_DIR / job_id / "metrics_user.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get(
    "/api/evaluations/{job_id}/report",
    response_class=HTMLResponse,
    responses={409: {"model": JobConflictResponse}},
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
        HTTPException 409: Job not yet complete; detail has shape
            {"job_id": ..., "status": "pending"|"running"|"error"}.
    """
    data = _read_status(job_id)
    if data["status"] != "ok":
        raise HTTPException(
            status_code=409,
            detail={"job_id": job_id, "status": data["status"]},
        )
    report_path = RUNS_DIR / job_id / "report.html"
    return report_path.read_text(encoding="utf-8")
