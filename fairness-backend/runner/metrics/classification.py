"""Classification-specific fairness metric functions.

These are passed to Fairlearn's MetricFrame as the `metrics` argument.
Each function has the signature (y_true, y_pred) -> float and is suitable
for per-group evaluation by MetricFrame.

selection_rate_ratio is computed post-hoc from per-group selection rates
(it is not a MetricFrame metric function).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def selection_rate(
    y_true: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """Fraction of samples predicted positive.

    Args:
        y_true: Ground-truth labels (unused; kept for MetricFrame compatibility).
        y_pred: Predicted labels.

    Returns:
        Selection rate in [0.0, 1.0].
    """
    return float(np.mean(np.asarray(y_pred) == 1))


def safe_true_positive_rate(
    y_true: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """TPR = TP / (TP + FN). Returns NaN when the group has no positive examples.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        TPR in [0.0, 1.0], or float('nan') if no positives in the group.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt == 1
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(yp[mask] == 1))


def safe_false_negative_rate(
    y_true: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """FNR = FN / (TP + FN). Returns NaN when the group has no positive examples.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        FNR in [0.0, 1.0], or float('nan') if no positives in the group.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt == 1
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(yp[mask] == 0))


def safe_false_positive_rate(
    y_true: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
) -> float:
    """FPR = FP / (FP + TN). Returns NaN when the group has no negative examples.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        FPR in [0.0, 1.0], or float('nan') if no negatives in the group.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt == 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(yp[mask] == 1))


def compute_selection_rate_ratio(rates_by_group: dict[str, float]) -> float | None:
    """Compute selection_rate_ratio from per-group selection rates.

    SRR = min(rates) / max(rates). Returns 0.0 if max(rates) == 0 (all
    groups predict negative). Returns None if fewer than 2 groups are present.
    The caller is responsible for emitting warnings based on which branch fired.

    Args:
        rates_by_group: Mapping from group label to its selection rate.

    Returns:
        Ratio in [0.0, 1.0], or None if the ratio is undefined.
    """
    rates = list(rates_by_group.values())
    if len(rates) < 2:
        return None
    max_rate = max(rates)
    if max_rate == 0.0:
        return 0.0
    return float(min(rates) / max_rate)
