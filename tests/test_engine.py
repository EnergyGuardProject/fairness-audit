"""Tests for FairnessEngine.

Covers: basic binary metrics, multiclass compatibility guard, intersectional
grouping, and three edge cases that arise from the synthetic dataset:
groups with zero positive examples, all-negative model predictions, and
a degenerate single-group configuration.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from runner.config.models import FairnessConfig, FairnessMetric
from runner.engine import EngineWarning, FairnessEngine, FairnessResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cfg(
    features: list[str],
    metrics: list[FairnessMetric] | None = None,
    intersectional: bool = False,
) -> FairnessConfig:
    if metrics is None:
        metrics = [
            FairnessMetric.demographic_parity_difference,
            FairnessMetric.equalized_odds_difference,
            FairnessMetric.selection_rate_ratio,
        ]
    return FairnessConfig(
        sensitive_features=features,
        fairness_metrics=metrics,
        parity_thresholds={},
        intersectional=intersectional,
    )


def _run(
    y_true: list[int],
    y_pred: list[int],
    feature_dict: dict[str, list],
    metrics: list[FairnessMetric] | None = None,
    intersectional: bool = False,
) -> FairnessResult:
    feature_names = list(feature_dict.keys())
    engine = FairnessEngine(_cfg(feature_names, metrics, intersectional))
    return engine.run(
        pd.Series(y_true),
        pd.Series(y_pred),
        pd.DataFrame(feature_dict),
        n_classes=2,
    )


# ── Existing tests (implemented) ──────────────────────────────────────────────

def test_binary_classification_metrics_computed() -> None:
    """Engine returns correct metric values for a simple binary case."""
    # Group A: 2 TP, 1 FP, 1 FN  →  acc=0.5, sel=0.5, tpr=0.5, fpr=0.5
    # Group B: 2 TP, 2 TN        →  acc=1.0, sel=0.5, tpr=1.0, fpr=0.0
    y_true = [0, 0, 1, 1,  0, 0, 1, 1]
    y_pred = [0, 1, 0, 1,  0, 0, 1, 1]
    result = _run(y_true, y_pred, {"grp": ["A","A","A","A","B","B","B","B"]})

    assert result.n_samples == 8
    assert result.n_classes == 2
    assert 0.0 <= result.overall_accuracy <= 1.0

    gA = result.by_group["grp=A"]
    gB = result.by_group["grp=B"]

    assert gA.accuracy == pytest.approx(0.5)
    assert gB.accuracy == pytest.approx(1.0)
    assert gA.true_positive_rate == pytest.approx(0.5)
    assert gB.true_positive_rate == pytest.approx(1.0)
    assert gA.false_positive_rate == pytest.approx(0.5)
    assert gB.false_positive_rate == pytest.approx(0.0)

    keys = {m.key for m in result.primary_metrics}
    assert "demographic_parity_difference" in keys
    assert "equalized_odds_difference" in keys
    assert "selection_rate_ratio" in keys

    # DPD: both groups have sel=0.5 → 0.0
    dpd = next(m for m in result.primary_metrics if m.key == "demographic_parity_difference")
    assert dpd.value == pytest.approx(0.0)

    # EOD: max(|0.5-0.0|, |0.5-0.0|) = 0.5
    eod = next(m for m in result.primary_metrics if m.key == "equalized_odds_difference")
    assert eod.value == pytest.approx(0.5)


def test_equalized_odds_incompatible_with_multiclass() -> None:
    """Engine raises ValueError when EOD is requested with n_classes > 2."""
    cfg = _cfg(["grp"], metrics=[FairnessMetric.equalized_odds_difference])
    engine = FairnessEngine(cfg)
    with pytest.raises(ValueError, match="n_classes"):
        engine.run(
            pd.Series([0, 1, 2, 0]),
            pd.Series([0, 1, 2, 0]),
            pd.DataFrame({"grp": ["A", "A", "B", "B"]}),
            n_classes=3,
        )


def test_intersectional_produces_cross_product_groups() -> None:
    """With intersectional=True, by_group keys are cross-product of features."""
    # 4 combinations: IT×1, IT×2, ES×1, ES×2 — each with 2 samples
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1]  # perfect predictions
    result = _run(
        y_true, y_pred,
        {
            "country": ["IT", "IT", "ES", "ES", "IT", "IT", "ES", "ES"],
            "decile":  [1,    1,    1,    1,    2,    2,    2,    2   ],
        },
        intersectional=True,
    )

    assert len(result.by_group) == 4
    keys = set(result.by_group.keys())
    assert any("IT" in k and "1" in k for k in keys)
    assert any("ES" in k and "2" in k for k in keys)
    # All groups have perfect accuracy
    for gm in result.by_group.values():
        assert gm.accuracy == pytest.approx(1.0)


def test_selection_rate_ratio_zero_max_rate_returns_zero() -> None:
    """SRR returns 0.0 without raising when max selection rate is zero."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 0]  # all-negative predictions
    result = _run(
        y_true, y_pred,
        {"grp": ["A", "A", "B", "B"]},
        metrics=[FairnessMetric.selection_rate_ratio],
    )
    srr = next(m for m in result.primary_metrics if m.key == "selection_rate_ratio")
    assert srr.value == pytest.approx(0.0)


def test_all_same_prediction_no_division_error() -> None:
    """No ZeroDivisionError when model predicts the same class for all samples."""
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [1, 1, 1, 1, 1, 1]  # all-positive predictions
    result = _run(y_true, y_pred, {"grp": ["A", "A", "A", "B", "B", "B"]})
    # Both groups predict all positive → selection_rate_ratio = 1.0
    srr = next(m for m in result.primary_metrics if m.key == "selection_rate_ratio")
    assert srr.value == pytest.approx(1.0)


# ── New edge-case tests ────────────────────────────────────────────────────────

def test_zero_positive_group_tpr_fnr_none_accuracy_computed() -> None:
    """Group with zero positives: TPR/FNR are None, accuracy and FPR are computed."""
    # neg_only: y_true = [0, 0]  — no positive examples
    # has_pos:  y_true = [0, 1]  — one positive, one negative
    y_true = [0, 0,  0, 1]
    y_pred = [0, 1,  0, 1]
    result = _run(y_true, y_pred, {"grp": ["neg_only", "neg_only", "has_pos", "has_pos"]})

    neg = result.by_group["grp=neg_only"]
    pos = result.by_group["grp=has_pos"]

    # Zero-positive group: TPR and FNR are undefined
    assert neg.true_positive_rate is None
    assert neg.false_negative_rate is None
    # FPR is still defined (there are negatives)
    assert neg.false_positive_rate is not None
    assert neg.false_positive_rate == pytest.approx(0.5)
    # Accuracy is always defined
    assert neg.accuracy == pytest.approx(0.5)

    # Group with positives: all binary metrics defined
    assert pos.true_positive_rate is not None
    assert pos.false_negative_rate is not None
    assert pos.true_positive_rate == pytest.approx(1.0)
    assert pos.false_negative_rate == pytest.approx(0.0)


def test_all_negative_predictions_srr_zero_with_warning() -> None:
    """All-zero model predictions: SRR == 0.0 and ALL_NEGATIVE_PREDICTIONS warning emitted."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 0]  # model never predicts positive
    result = _run(
        y_true, y_pred,
        {"grp": ["A", "A", "B", "B"]},
        metrics=[FairnessMetric.selection_rate_ratio],
    )

    srr = next(m for m in result.primary_metrics if m.key == "selection_rate_ratio")
    assert srr.value == pytest.approx(0.0)

    codes = {w.code for w in result.warnings}
    assert "ALL_NEGATIVE_PREDICTIONS" in codes


def test_single_group_aggregated_metric_none_with_warning() -> None:
    """Only one group present: aggregated metric is None with INSUFFICIENT_GROUPS warning."""
    y_true = [0, 1]
    y_pred = [1, 0]
    sensitive = pd.DataFrame({"grp": ["only", "only"]})
    engine = FairnessEngine(_cfg(["grp"]))
    result = engine.run(pd.Series(y_true), pd.Series(y_pred), sensitive, n_classes=2)

    dpd = next(m for m in result.primary_metrics if m.key == "demographic_parity_difference")
    assert dpd.value is None

    codes = {w.code for w in result.warnings}
    assert "INSUFFICIENT_GROUPS_FOR_METRIC" in codes
