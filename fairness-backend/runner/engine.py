"""Fairness computation engine wrapping Fairlearn's MetricFrame.

Pure compute layer — no I/O, no Plotly, no HTML. Accepts arrays/DataFrames
and returns a FairnessResult Pydantic model.

Key design decisions:
- n_classes is inferred upstream (pipeline.py) from y_true.nunique() and passed
  in explicitly so the engine never touches the raw DataFrame.
- Metric compatibility is checked at entry before any computation.
- selection_rate_ratio is derived from per-group selection rates post-hoc
  (not a native Fairlearn function).
- intersectional=True: sensitive features are combined into a cross-product
  string label before grouping.
- Groups with zero positive examples return None for TPR/FNR (undefined);
  FPR and accuracy are still computed.
- Aggregated metrics with fewer than 2 valid groups emit a warning and
  return None rather than raising.
"""
from __future__ import annotations

import logging

import numpy as np
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


class EngineWarning(BaseModel):
    """A warning emitted during metric computation."""

    code: str
    message: str


class GroupMetrics(BaseModel):
    """Per-group metric values for a single sensitive feature group."""

    accuracy: float
    selection_rate: float
    true_positive_rate: float | None   # None if group has no positive examples, or multiclass
    false_positive_rate: float | None  # None if group has no negative examples, or multiclass
    false_negative_rate: float | None  # None if group has no positive examples, or multiclass
    n_samples: int


class PrimaryMetricValue(BaseModel):
    """A single thresholded primary metric value."""

    key: str
    value: float | None  # None when metric cannot be computed (e.g. insufficient groups)


class FairnessResult(BaseModel):
    """Output of FairnessEngine.run(). Pure data — no threshold evaluation."""

    primary_metrics: list[PrimaryMetricValue]
    overall_accuracy: float
    overall_selection_rate: float
    by_group: dict[str, GroupMetrics]  # key: "feature=value" or "f1=v1,f2=v2"
    n_samples: int
    n_classes: int
    warnings: list[EngineWarning]


class FairnessEngine:
    """Compute group fairness metrics for a classification evaluation.

    Metrics are computed directly with NumPy (no Fairlearn MetricFrame
    dependency at runtime).  Sensitive features are iterated individually;
    when ``config.intersectional=True`` they are first combined into a single
    cross-product label column.
    """

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
            FairnessResult with all primary metric values, per-group
            breakdowns, and any warnings emitted during computation.

        Raises:
            ValueError: If any requested metric is incompatible with n_classes
                (e.g., equalized_odds_difference with n_classes > 2). The error
                message lists all incompatible metrics so the caller can surface
                them to the user in one shot.
        """
        self._check_compatibility(n_classes)

        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        warnings_list: list[EngineWarning] = []
        by_group: dict[str, GroupMetrics] = {}

        # Build (feature_name, series) pairs for grouping
        feature_cols = sensitive.columns.tolist()
        if self._config.intersectional and len(feature_cols) > 1:
            combined = sensitive.apply(
                lambda row: ",".join(f"{c}={row[c]}" for c in feature_cols),
                axis=1,
            )
            feature_sets = [("intersectional", combined)]
        else:
            feature_sets = [(col, sensitive[col]) for col in feature_cols]

        per_feature_results: list[dict[str, float | None]] = []

        for feature_name, sf in feature_sets:
            group_values = sf.unique()

            per_group_sel: dict[str, float] = {}
            per_group_tpr: dict[str, float | None] = {}
            per_group_fpr: dict[str, float | None] = {}
            per_group_fnr: dict[str, float | None] = {}

            for gv in group_values:
                mask = np.asarray(sf == gv)
                yt_g = yt[mask]
                yp_g = yp[mask]

                n_pos = int((yt_g == 1).sum())
                n_neg = len(yt_g) - n_pos

                acc = float(np.mean(yt_g == yp_g))
                sel = float(np.mean(yp_g == 1))
                tpr = float(np.mean(yp_g[yt_g == 1] == 1)) if n_pos > 0 else None
                fnr = float(np.mean(yp_g[yt_g == 1] == 0)) if n_pos > 0 else None
                fpr = float(np.mean(yp_g[yt_g == 0] == 1)) if n_neg > 0 else None

                key = str(gv) if feature_name == "intersectional" else f"{feature_name}={gv}"
                by_group[key] = GroupMetrics(
                    accuracy=acc,
                    selection_rate=sel,
                    true_positive_rate=tpr,
                    false_positive_rate=fpr,
                    false_negative_rate=fnr,
                    n_samples=int(mask.sum()),
                )

                g_key = str(gv)
                per_group_sel[g_key] = sel
                per_group_tpr[g_key] = tpr
                per_group_fpr[g_key] = fpr
                per_group_fnr[g_key] = fnr

            feature_result = self._compute_feature_metrics(
                feature_name, per_group_sel, per_group_tpr, per_group_fpr,
                per_group_fnr, warnings_list,
            )
            per_feature_results.append(feature_result)

        # Aggregate across features: worst case
        primary_metrics: list[PrimaryMetricValue] = []
        for metric in self._config.fairness_metrics:
            mkey = metric.value
            values = [
                fr[mkey] for fr in per_feature_results
                if fr.get(mkey) is not None
            ]
            if not values:
                agg: float | None = None
            elif mkey == "selection_rate_ratio":
                agg = float(min(values))  # worst = lowest ratio
            else:
                agg = float(max(values))  # worst = highest difference
            primary_metrics.append(PrimaryMetricValue(key=mkey, value=agg))

        return FairnessResult(
            primary_metrics=primary_metrics,
            overall_accuracy=float(np.mean(yt == yp)),
            overall_selection_rate=float(np.mean(yp == 1)),
            by_group=by_group,
            n_samples=len(yt),
            n_classes=n_classes,
            warnings=warnings_list,
        )

    def _compute_feature_metrics(
        self,
        feature_name: str,
        per_group_sel: dict[str, float],
        per_group_tpr: dict[str, float | None],
        per_group_fpr: dict[str, float | None],
        per_group_fnr: dict[str, float | None],
        warnings_list: list[EngineWarning],
    ) -> dict[str, float | None]:
        """Compute all requested primary metrics for a single sensitive feature.

        Args:
            feature_name: Name of the sensitive feature (for warning messages).
            per_group_sel: Selection rates keyed by group value string.
            per_group_tpr: TPR (or None) keyed by group value string.
            per_group_fpr: FPR (or None) keyed by group value string.
            per_group_fnr: FNR (or None) keyed by group value string.
            warnings_list: Mutable list to append EngineWarning objects to.

        Returns:
            Dict mapping metric key → computed value (None if undefined).
        """
        result: dict[str, float | None] = {}
        rates = list(per_group_sel.values())
        valid_fpr = {k: v for k, v in per_group_fpr.items() if v is not None}
        valid_fnr = {k: v for k, v in per_group_fnr.items() if v is not None}

        for metric in self._config.fairness_metrics:
            mkey = metric.value

            if mkey == "demographic_parity_difference":
                if len(rates) >= 2:
                    result[mkey] = float(max(rates) - min(rates))
                else:
                    warnings_list.append(EngineWarning(
                        code="INSUFFICIENT_GROUPS_FOR_METRIC",
                        message=(
                            f"Feature '{feature_name}': fewer than 2 groups "
                            f"for {mkey}."
                        ),
                    ))
                    result[mkey] = None

            elif mkey == "equalized_odds_difference":
                fpr_diff = (
                    float(max(valid_fpr.values()) - min(valid_fpr.values()))
                    if len(valid_fpr) >= 2 else None
                )
                fnr_diff = (
                    float(max(valid_fnr.values()) - min(valid_fnr.values()))
                    if len(valid_fnr) >= 2 else None
                )
                candidates = [v for v in (fpr_diff, fnr_diff) if v is not None]
                if candidates:
                    result[mkey] = float(max(candidates))
                else:
                    warnings_list.append(EngineWarning(
                        code="INSUFFICIENT_GROUPS_FOR_METRIC",
                        message=(
                            f"Feature '{feature_name}': fewer than 2 groups with "
                            f"defined FPR or FNR for {mkey}."
                        ),
                    ))
                    result[mkey] = None

            elif mkey == "selection_rate_ratio":
                max_rate = max(rates) if rates else 0.0
                if max_rate == 0.0:
                    warnings_list.append(EngineWarning(
                        code="ALL_NEGATIVE_PREDICTIONS",
                        message=(
                            f"Feature '{feature_name}': all groups have zero "
                            f"selection rate; SRR set to 0.0."
                        ),
                    ))
                    result[mkey] = 0.0
                elif len(rates) >= 2:
                    result[mkey] = float(min(rates) / max_rate)
                else:
                    warnings_list.append(EngineWarning(
                        code="INSUFFICIENT_GROUPS_FOR_METRIC",
                        message=(
                            f"Feature '{feature_name}': fewer than 2 groups "
                            f"for {mkey}."
                        ),
                    ))
                    result[mkey] = None

            elif mkey == "false_positive_rate_difference":
                if len(valid_fpr) >= 2:
                    result[mkey] = float(max(valid_fpr.values()) - min(valid_fpr.values()))
                else:
                    warnings_list.append(EngineWarning(
                        code="INSUFFICIENT_GROUPS_FOR_METRIC",
                        message=(
                            f"Feature '{feature_name}': fewer than 2 groups with "
                            f"defined FPR for {mkey}."
                        ),
                    ))
                    result[mkey] = None

            elif mkey == "false_negative_rate_difference":
                if len(valid_fnr) >= 2:
                    result[mkey] = float(max(valid_fnr.values()) - min(valid_fnr.values()))
                else:
                    warnings_list.append(EngineWarning(
                        code="INSUFFICIENT_GROUPS_FOR_METRIC",
                        message=(
                            f"Feature '{feature_name}': fewer than 2 groups with "
                            f"defined FNR for {mkey}."
                        ),
                    ))
                    result[mkey] = None

        return result

    def _check_compatibility(self, n_classes: int) -> None:
        """Raise ValueError if any configured metric is incompatible with n_classes.

        Args:
            n_classes: Number of distinct label classes.

        Raises:
            ValueError: Lists all incompatible metrics in the message.
        """
        if n_classes > 2:
            bad = [
                m.value for m in self._config.fairness_metrics
                if m in _BINARY_ONLY_METRICS
            ]
            if bad:
                raise ValueError(
                    f"The following metrics require binary classification "
                    f"(n_classes == 2) but n_classes={n_classes}: {bad}"
                )

