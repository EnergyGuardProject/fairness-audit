"""Classification-specific fairness metric functions.

These are passed to Fairlearn's MetricFrame as the `metrics` argument.
Each function has the signature (y_true, y_pred) → float and is suitable
for per-group evaluation by MetricFrame.

selection_rate_ratio is computed post-hoc from the MetricFrame's per-group
selection rates (it is not passed as a MetricFrame metric function).
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
        y_true: Ground-truth labels (unused; signature matches MetricFrame
            convention).
        y_pred: Predicted labels.

    Returns:
        Selection rate in [0.0, 1.0].
    """
    raise NotImplementedError


def compute_selection_rate_ratio(rates_by_group: dict[str, float]) -> float:
    """Compute selection_rate_ratio from per-group selection rates.

    SRR = min(rates) / max(rates). Returns 0.0 and logs a warning if
    max(rates) == 0 (no positive predictions anywhere).

    Args:
        rates_by_group: Dict mapping group label to its selection rate.

    Returns:
        Ratio in [0.0, 1.0].
    """
    raise NotImplementedError
