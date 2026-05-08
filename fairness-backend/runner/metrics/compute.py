"""Build the full metrics.json firehose output.

Includes all numeric values from FairnessResult plus a provenance block
(library versions, config hash, git SHA).
"""
from __future__ import annotations

import logging
from typing import Any

from runner.config.models import RunConfig
from runner.engine import FairnessResult

logger = logging.getLogger(__name__)


def build_metrics_full(
    result: FairnessResult,
    config: RunConfig,
    job_id: str,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Build the metrics.json firehose payload.

    Args:
        result: Output of FairnessEngine.run().
        config: The RunConfig for this evaluation.
        job_id: Job identifier.
        timestamp_utc: ISO-8601 UTC timestamp string.

    Returns:
        Dict with provenance block, all metric values, and per-group breakdown.
    """
    raise NotImplementedError


def _build_provenance(config: RunConfig) -> dict[str, Any]:
    """Build the provenance block for metrics.json.

    Includes library versions, config hash (SHA-256 of sorted JSON), and
    git SHA (best-effort from `git rev-parse HEAD`; null if unavailable).

    Args:
        config: RunConfig to hash.

    Returns:
        Dict with version strings, config_hash, and git_sha.
    """
    raise NotImplementedError
