"""Fairness computation engine wrapping Fairlearn's MetricFrame.

Pure compute layer — no I/O, no Plotly, no HTML. Accepts arrays/DataFrames
and returns a FairnessResult Pydantic model.

Key design decisions:
- n_classes is inferred upstream (pipeline.py) from y_true.nunique() and passed
  in explicitly so the engine never touches the raw DataFrame.
- Metric compatibility is checked at entry before any computation.
- selection_rate_ratio is derived from MetricFrame per-group selection rates
  (not a native Fairlearn function).
- intersectional=True: sensitive features passed as a DataFrame to MetricFrame,
  producing cross-product group labels.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel

from runner.config.models import FairnessConfig, FairnessMetric

logger = logging.getLogger(__name__)

# Metrics that require binary classification (n_classes == 2)
_BINARY_ONLY_METRICS = frozenset({
    FairnessMetric.equalized_odds_difference,
    FairnessMetric.false_positive_rate_difference,
    FairnessMetric.false_negative_rate_difference,
})


class GroupMetrics(BaseModel):
    """Per-group metric values for a single sensitive feature group."""

    accuracy: float
    selection_rate: float
    false_positive_rate: float | None  # None for multiclass
    false_negative_rate: float | None  # None for multiclass
    n_samples: int


class PrimaryMetricValue(BaseModel):
    """A single thresholded primary metric value."""

    key: str
    value: float


class FairnessResult(BaseModel):
    """Output of FairnessEngine.run(). Pure data — no threshold evaluation."""

    primary_metrics: list[PrimaryMetricValue]
    overall_accuracy: float
    overall_selection_rate: float
    by_group: dict[str, GroupMetrics]  # key: "feature=value" or "f1=v1,f2=v2"
    n_samples: int
    n_classes: int


class FairnessEngine:
    """Wraps Fairlearn MetricFrame for fairness metric computation."""

    def __init__(self, config: FairnessConfig) -> None:
        """Store config and build internal metric function registry.

        Args:
            config: FairnessConfig specifying metrics, thresholds, and
                sensitive feature names.
        """
        self._config = config

    def run(
        self,
        y_true: pd.Series,  # type: ignore[type-arg]
        y_pred: pd.Series,  # type: ignore[type-arg]
        sensitive: pd.DataFrame,
        n_classes: int,
    ) -> FairnessResult:
        """Compute fairness metrics for a classification evaluation.

        Args:
            y_true: Ground-truth labels.
            y_pred: Model predictions (hard labels, not probabilities).
            sensitive: DataFrame of sensitive feature columns. Column names
                must match config.sensitive_features.
            n_classes: Number of distinct classes (from y_true.nunique()).
                Passed explicitly so the engine has no dependency on the
                raw dataset.

        Returns:
            FairnessResult with all primary metric values and per-group
            breakdowns.

        Raises:
            ValueError: If any requested metric is incompatible with n_classes
                (e.g., equalized_odds_difference with n_classes > 2). The error
                message lists all incompatible metrics so the caller can surface
                them to the user in one shot.
        """
        raise NotImplementedError

    def _check_compatibility(self, n_classes: int) -> None:
        """Raise ValueError if any configured metric is incompatible with n_classes.

        Args:
            n_classes: Number of distinct label classes.

        Raises:
            ValueError: Lists all incompatible metrics in the message.
        """
        raise NotImplementedError

    def _build_metric_functions(self) -> dict[str, Any]:
        """Return a dict of metric_name → callable for MetricFrame.

        Returns:
            Dict mapping metric key strings to Fairlearn/sklearn callables.
        """
        raise NotImplementedError
