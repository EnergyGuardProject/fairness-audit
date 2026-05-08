"""Orchestrates the end-to-end fairness audit pipeline.

Sequence:
    1. Load and validate RunConfig from YAML
    2. Generate job_id; create job output directory
    3. Write config_resolved.yaml + configure job-scoped file logger
    4. Select TaskStrategy (v1: always ClassificationFairnessStrategy)
    5. strategy.execute(config, job_dir)  →  writes all artifacts

Called by both the CLI (runner/cli.py) and the FastAPI background task
(app/main.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from runner.config.models import RunConfig

logger = logging.getLogger(__name__)


def run(
    config_path: Path,
    output_root: Path = Path("runs"),
    job_id_override: str | None = None,
) -> str:
    """Load config, create job directory, execute strategy, return job_id.

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
    raise NotImplementedError


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
    raise NotImplementedError


def _generate_job_id() -> str:
    """Generate a time-stamped job identifier.

    Returns:
        String of the form YYYYMMDD_HHMMSS_<8-char hex>.
    """
    raise NotImplementedError


def _setup_job_logger(job_dir: Path) -> logging.FileHandler:
    """Add a file handler to the root logger writing to job_dir/run.log.

    Args:
        job_dir: Directory for this job's artifacts.

    Returns:
        The FileHandler (caller should remove it after the job completes).
    """
    raise NotImplementedError
