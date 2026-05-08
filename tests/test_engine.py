"""Tests for FairnessEngine.

These tests require numpy and pandas but not the full fairlearn stack.
They will be fully implemented in the engine implementation session.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="FairnessEngine not yet implemented — scaffold only")
def test_binary_classification_metrics_computed() -> None:
    """Engine returns correct metric values for a simple binary case."""
    pass


@pytest.mark.skip(reason="FairnessEngine not yet implemented — scaffold only")
def test_equalized_odds_incompatible_with_multiclass() -> None:
    """Engine raises ValueError when EOD is requested with n_classes > 2."""
    pass


@pytest.mark.skip(reason="FairnessEngine not yet implemented — scaffold only")
def test_intersectional_produces_cross_product_groups() -> None:
    """With intersectional=True, groups are cross-product of sensitive features."""
    pass


@pytest.mark.skip(reason="FairnessEngine not yet implemented — scaffold only")
def test_selection_rate_ratio_zero_max_rate_returns_zero() -> None:
    """SRR returns 0.0 without raising when max selection rate is zero."""
    pass


@pytest.mark.skip(reason="FairnessEngine not yet implemented — scaffold only")
def test_all_same_prediction_no_division_error() -> None:
    """No ZeroDivisionError when model predicts the same class for all samples."""
    pass
