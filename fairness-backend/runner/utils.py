"""Shared utilities for the fairness audit runner.

Extracted here to avoid duplication between runner/pipeline.py and app/main.py.
Add a third caller before creating a new module; keep this file small.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_job_id() -> str:
    """Generate a time-stamped job identifier.

    Returns:
        String of the form YYYYMMDD_HHMMSS_<8-char hex>.
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def write_status(job_dir: Path, payload: dict[str, Any]) -> None:
    """Write status.json to job_dir atomically (overwrite if exists).

    Args:
        job_dir: Job output directory.
        payload: Status dict to serialise as JSON.
    """
    (job_dir / "status.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
