"""Orchestrates the end-to-end fairness audit pipeline.

Sequence (run):
    1. Load and validate RunConfig from YAML
    2. Generate job_id; create job output directory
    3. Configure job-scoped file logger (run.log)
    4. Call run_evaluation(config, job_dir, job_id, config_path)
    5. Return job_id

Sequence (run_evaluation):
    1. Write config.yaml (raw) and config_resolved.yaml (absolute paths)
    2. Write status.json {"status": "running", ...}
    3. Load model and dataset via loaders
    4. Compute predictions via ModelAdapter
    5. Infer n_classes from y_true.nunique()
    6. Run FairnessEngine.run()
    7. Build provenance dict
    8. Write metrics.json (firehose)
    9. Build and write metrics_user.json
    10. Render and write report.html
    11. Write status.json {"status": "ok", ...}
    On exception: log, write status.json {"status": "error", ...}, re-raise

Called by both the CLI (runner/cli.py) and the FastAPI background task
(app/main.py).

NOTE — v1 TaskStrategy bypass:
    ClassificationFairnessStrategy in task_strategy.py is intentionally left
    as a stub. The task-strategy dispatch was designed for future extensibility
    (regression in v1.1, subgroup-performance in v1.2), but v1 has exactly one
    task type. The compute logic lives inline in run_evaluation() to avoid
    premature indirection. Wire up TaskStrategy dispatch in v1.1 when a second
    strategy is needed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from runner.config.models import RunConfig

logger = logging.getLogger(__name__)

try:
    _SERVICE_VERSION = importlib.metadata.version("fairness-audit")
except importlib.metadata.PackageNotFoundError:
    _SERVICE_VERSION = "0.1.0-dev"


def run(
    config_path: Path,
    output_root: Path = Path("runs"),
    job_id_override: str | None = None,
) -> str:
    """Load config, create job directory, execute evaluation, return job_id.

    Args:
        config_path: Path to the YAML RunConfig file.
        output_root: Parent directory for job output directories.
        job_id_override: If provided, use this string as the job_id instead of
            generating one. Useful for determinism tests.

    Returns:
        The job_id string (also the name of the created subdirectory under
        output_root).

    Raises:
        pydantic.ValidationError: If the YAML config is invalid.
        ValueError: If the dataset or model cannot be loaded, or if the
            requested metrics are incompatible with the task type.
    """
    config = _load_config(config_path)
    job_id = job_id_override or _generate_job_id()
    job_dir = Path(output_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    handler = _setup_job_logger(job_dir)
    try:
        run_evaluation(config, job_dir, job_id, config_path)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()

    return job_id


def run_evaluation(
    config: RunConfig,
    job_dir: Path,
    job_id: str,
    config_path: Path | None = None,
) -> None:
    """Execute the full fairness evaluation pipeline and write all artifacts.

    Args:
        config: Validated RunConfig.
        job_dir: Output directory for this job's artifacts.
        job_id: Job identifier string (used as run_id in report_meta).
        config_path: Path to the original config file (copied verbatim as
            config.yaml). If None, config.yaml is written from the serialised
            RunConfig dict.

    Raises:
        Exception: Any error is logged and written to status.json, then
            re-raised so the caller (CLI or FastAPI) can handle it.
    """
    import pandas as pd

    from runner.engine import FairnessEngine
    from runner.loaders.dataset_loaders import load_csv_dataset
    from runner.loaders.model_loaders import load_sklearn_model
    from runner.metrics.firehose import write_metrics_json
    from runner.metrics.user_metrics import build_metrics_user
    from runner.reporting.html_report import render_html_report

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Write config.yaml (raw) and config_resolved.yaml (absolute paths)
    if config_path is not None:
        (job_dir / "config.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        (job_dir / "config.yaml").write_text(
            yaml.dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    resolved_dict = config.model_dump(mode="json")
    resolved_dict["dataset"]["path"] = str(Path(config.dataset.path).resolve())
    resolved_dict["model"]["path"] = str(Path(config.model.path).resolve())
    (job_dir / "config_resolved.yaml").write_text(
        yaml.dump(resolved_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # 2. Write running status
    _write_status(job_dir, {"status": "running", "created_at": created_at})

    try:
        # 3. Load model and dataset
        logger.info("Loading model from %s", config.model.path)
        adapter = load_sklearn_model(Path(config.model.path))

        logger.info("Loading dataset from %s", config.dataset.path)
        dataset = load_csv_dataset(config.dataset, config.fairness.sensitive_features)

        # 4. Compute predictions
        logger.info("Running predictions on %d samples", len(dataset.X))
        y_pred_arr = adapter.predict(dataset.X)
        y_pred = pd.Series(y_pred_arr, index=dataset.y_true.index)

        # 5. Infer n_classes
        n_classes = int(dataset.y_true.nunique())
        logger.info("Detected %d classes in y_true", n_classes)

        # 6. Run fairness engine
        engine = FairnessEngine(config.fairness)
        logger.info("Running FairnessEngine")
        result = engine.run(
            y_true=dataset.y_true,
            y_pred=y_pred,
            sensitive=dataset.sensitive,
            n_classes=n_classes,
        )
        logger.info(
            "Engine complete: %d groups, %d warnings",
            len(result.by_group), len(result.warnings),
        )

        # 7. Build provenance
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        config_hash = hashlib.sha256(
            json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()

        try:
            git_sha: str | None = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_sha = None

        provenance: dict[str, Any] = {
            "job_id": job_id,
            "timestamp_utc": timestamp_utc,
            "service_version": _SERVICE_VERSION,
            "fairlearn_version": importlib.metadata.version("fairlearn"),
            "sklearn_version": importlib.metadata.version("scikit-learn"),
            "numpy_version": importlib.metadata.version("numpy"),
            "pandas_version": importlib.metadata.version("pandas"),
            "config_hash": config_hash,
            "git_sha": git_sha,
            "feature_count": len(dataset.X.columns),
        }

        # 8. Write metrics.json (firehose)
        write_metrics_json(result, provenance, job_dir / "metrics.json")

        # 9. Build and write metrics_user.json
        metrics_user = build_metrics_user(result, config, provenance)
        (job_dir / "metrics_user.json").write_text(
            json.dumps(metrics_user, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Written metrics_user.json")

        # 10. Render and write report.html
        render_html_report(result, metrics_user, config, job_dir / "report.html")

        # 11. Write OK status
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status(job_dir, {
            "status": "ok",
            "created_at": created_at,
            "completed_at": completed_at,
        })
        logger.info("Job %s completed successfully", job_id)

    except Exception as exc:
        logger.error("Job %s failed: %s", exc, exc_info=True)
        completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status(job_dir, {
            "status": "error",
            "error": str(exc),
            "created_at": created_at,
            "completed_at": completed_at,
        })
        raise


def _write_status(job_dir: Path, payload: dict[str, Any]) -> None:
    """Write status.json to job_dir.

    Args:
        job_dir: Job output directory.
        payload: Status dict to serialise.
    """
    (job_dir / "status.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _load_config(config_path: Path) -> RunConfig:
    """Parse YAML and validate with Pydantic.

    Args:
        config_path: Path to the YAML RunConfig file.

    Returns:
        Validated RunConfig.

    Raises:
        FileNotFoundError: If config_path does not exist.
        pydantic.ValidationError: If the YAML content is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    return RunConfig.model_validate(raw)


def _generate_job_id() -> str:
    """Generate a time-stamped job identifier.

    Returns:
        String of the form YYYYMMDD_HHMMSS_<8-char hex>.
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _setup_job_logger(job_dir: Path) -> logging.FileHandler:
    """Add a file handler to the root logger writing to job_dir/run.log.

    Args:
        job_dir: Directory for this job's artifacts.

    Returns:
        The FileHandler (caller must remove it and close it after the job).
    """
    handler = logging.FileHandler(job_dir / "run.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler
