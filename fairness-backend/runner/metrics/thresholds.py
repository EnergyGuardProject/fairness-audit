"""Centralised threshold definitions and metric status evaluation.

All pass/warn/fail logic lives here. No threshold logic exists anywhere else
in the codebase.

Default boundaries:

    Difference metrics (lower_is_better):
        fail boundary: value > fail_threshold
        warn boundary: value > warn_threshold  (= fail / 2)

    Ratio metrics (higher_is_better):
        fail boundary: value < fail_threshold
        warn boundary: value < warn_threshold  (= (1.0 + fail) / 2)

    Rationale for ratio warn: applying "warn = fail / 2" to ratio metrics gives
    warn = 0.40 for fail = 0.80, which corresponds to a severely unfair model.
    Using the midpoint between 1.0 (perfect equality) and the fail boundary
    produces a warn zone proportional to the acceptable range (0.80..0.90).

User override: parity_thresholds in FairnessConfig replaces the fail boundary
for the specified metric(s). The warn boundary is always re-derived using the
same formula above, so only one value needs to be overridden.
"""
from __future__ import annotations

from typing import Literal, TypedDict


class ThresholdSpec(TypedDict):
    """Pass/warn/fail boundaries for a single fairness metric."""

    fail: float
    warn: float
    direction: Literal["lower_is_better", "higher_is_better"]


DEFAULT_THRESHOLDS: dict[str, ThresholdSpec] = {
    "demographic_parity_difference": ThresholdSpec(
        fail=0.10, warn=0.05, direction="lower_is_better"
    ),
    "equalized_odds_difference": ThresholdSpec(
        fail=0.10, warn=0.05, direction="lower_is_better"
    ),
    "selection_rate_ratio": ThresholdSpec(
        fail=0.80, warn=0.90, direction="higher_is_better"
    ),
}


def _derive_warn(
    fail: float,
    direction: Literal["lower_is_better", "higher_is_better"],
) -> float:
    """Compute the warn boundary from the fail boundary.

    Args:
        fail: The fail boundary value.
        direction: Metric direction.

    Returns:
        Derived warn boundary.
    """
    if direction == "lower_is_better":
        return fail / 2.0
    return (1.0 + fail) / 2.0


def resolve_thresholds(
    metric_key: str,
    parity_thresholds: dict[str, float],
) -> ThresholdSpec:
    """Return the effective ThresholdSpec for a metric, applying user overrides.

    Args:
        metric_key: Metric name (e.g. "demographic_parity_difference").
        parity_thresholds: User-supplied fail-boundary overrides from
            FairnessConfig. May be empty.

    Returns:
        ThresholdSpec with effective fail, warn, and direction values.

    Raises:
        KeyError: If metric_key is not in DEFAULT_THRESHOLDS (i.e., it is a
            diagnostic metric with no threshold).
    """
    base = DEFAULT_THRESHOLDS[metric_key]
    if metric_key not in parity_thresholds:
        return base
    new_fail = parity_thresholds[metric_key]
    new_warn = _derive_warn(new_fail, base["direction"])
    return ThresholdSpec(fail=new_fail, warn=new_warn, direction=base["direction"])


def evaluate_metric(
    key: str,
    value: float,
    parity_thresholds: dict[str, float],
) -> Literal["pass", "warn", "fail"]:
    """Classify a metric value as pass, warn, or fail.

    Args:
        key: Metric name. Must be a key in DEFAULT_THRESHOLDS.
        value: Computed metric value.
        parity_thresholds: User-supplied fail-boundary overrides.

    Returns:
        "pass", "warn", or "fail".

    Raises:
        KeyError: If key is not a known thresholded metric.
    """
    spec = resolve_thresholds(key, parity_thresholds)
    direction = spec["direction"]
    fail = spec["fail"]
    warn = spec["warn"]

    if direction == "lower_is_better":
        if value > fail:
            return "fail"
        if value > warn:
            return "warn"
        return "pass"
    else:  # higher_is_better
        if value < fail:
            return "fail"
        if value < warn:
            return "warn"
        return "pass"
