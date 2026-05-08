"""Firehose metrics.json writer — full FairnessResult dump + provenance block.

Kept separate from user_metrics.py so the schema-conforming projection
(metrics_user.json) remains independent of the raw firehose output.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from runner.engine import FairnessResult

logger = logging.getLogger(__name__)


def write_metrics_json(
    result: FairnessResult,
    provenance: dict[str, Any],
    path: Path,
) -> None:
    """Serialise FairnessResult + provenance block to metrics.json.

    The by_group dict is sorted by key before serialisation to ensure
    byte-stable output across runs.

    Args:
        result: Output of FairnessEngine.run().
        provenance: Dict containing library versions, config_hash, git_sha,
            timestamp_utc, job_id, service_version, and feature_count.
        path: Destination file path (written via write_text).

    Raises:
        ValueError: If the serialised payload fails its own JSON round-trip
            (catches non-serialisable values before touching disk).
        OSError: If path cannot be written.
    """
    payload = result.model_dump()

    # Sort by_group for determinism across runs
    if "by_group" in payload:
        payload["by_group"] = dict(sorted(payload["by_group"].items()))

    payload["provenance"] = provenance

    raw = json.dumps(payload, sort_keys=True, default=str)

    # Round-trip check — catches non-serialisable values with a clear traceback
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"metrics.json failed JSON round-trip check: {exc}"
        ) from exc

    path.write_text(raw, encoding="utf-8")
    logger.info("Written metrics.json to %s (%d bytes)", path, len(raw))
