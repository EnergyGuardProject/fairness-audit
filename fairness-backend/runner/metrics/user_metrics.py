"""Map FairnessResult → metrics_user.json payload (REPORT_SCHEMA.md v1.0).

Also implements headline score and label computation.

Scoring rubric (documented in docs/FAIRNESS_METRICS.md):
    pass → 1.0, warn → 0.5, fail → 0.0
    headline_score = mean(scores for primary metrics)
    headline_label: good (all pass) | moderate (warn only) | poor (any fail)
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from runner.config.models import RunConfig
from runner.engine import FairnessResult
from runner.metrics.thresholds import evaluate_metric

logger = logging.getLogger(__name__)

_STATUS_SCORE: dict[str, float] = {"pass": 1.0, "warn": 0.5, "fail": 0.0}

# Metrics that appear in summary.primary_metrics with pass/warn/fail status.
THRESHOLDED_METRICS = frozenset({
    "demographic_parity_difference",
    "equalized_odds_difference",
    "selection_rate_ratio",
})


def compute_headline_score(statuses: list[Literal["pass", "warn", "fail"]]) -> float:
    """Compute the headline score as the mean of per-metric status scores.

    Args:
        statuses: List of status strings for all primary metrics.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for an empty list.
    """
    if not statuses:
        return 0.0
    return sum(_STATUS_SCORE[s] for s in statuses) / len(statuses)


def derive_headline_label(
    statuses: list[Literal["pass", "warn", "fail"]],
) -> Literal["good", "moderate", "poor"]:
    """Derive the headline label from primary metric statuses.

    Label is derived from status composition, not the numeric score:
        good     — all pass
        moderate — at least one warn, no fails
        poor     — at least one fail

    Args:
        statuses: List of status strings for all primary metrics.

    Returns:
        "good", "moderate", or "poor".
    """
    if any(s == "fail" for s in statuses):
        return "poor"
    if any(s == "warn" for s in statuses):
        return "moderate"
    return "good"


def build_metrics_user(
    result: FairnessResult,
    config: RunConfig,
    job_id: str,
    timestamp_utc: str,
    service_version: str,
) -> dict[str, Any]:
    """Build the full metrics_user.json payload from a FairnessResult.

    Args:
        result: Output of FairnessEngine.run().
        config: The RunConfig used for this evaluation.
        job_id: Job identifier (used as report_meta.run_id).
        timestamp_utc: ISO-8601 UTC timestamp string.
        service_version: Version string from importlib.metadata.

    Returns:
        Dict conforming to docs/REPORT_SCHEMA.md v1.0.
    """
    raise NotImplementedError
