"""Tests for threshold evaluation and headline scoring.

These tests exercise runner/metrics/thresholds.py and the pure arithmetic
functions in runner/metrics/user_metrics.py. No fairlearn or sklearn needed.
"""
from __future__ import annotations

import pytest

from runner.metrics.thresholds import (
    DEFAULT_THRESHOLDS,
    ThresholdSpec,
    _derive_warn,
    evaluate_metric,
    resolve_thresholds,
)
from runner.metrics.user_metrics import compute_headline_score, derive_headline_label


# ── _derive_warn ───────────────────────────────────────────────────────────────

def test_derive_warn_lower_is_better() -> None:
    assert _derive_warn(0.10, "lower_is_better") == pytest.approx(0.05)


def test_derive_warn_higher_is_better() -> None:
    # fail=0.80 → warn=(1.0+0.80)/2=0.90
    assert _derive_warn(0.80, "higher_is_better") == pytest.approx(0.90)


def test_derive_warn_higher_is_better_custom() -> None:
    # fail=0.60 → warn=0.80
    assert _derive_warn(0.60, "higher_is_better") == pytest.approx(0.80)


# ── DEFAULT_THRESHOLDS sanity ──────────────────────────────────────────────────

def test_default_thresholds_keys_present() -> None:
    assert "demographic_parity_difference" in DEFAULT_THRESHOLDS
    assert "equalized_odds_difference" in DEFAULT_THRESHOLDS
    assert "selection_rate_ratio" in DEFAULT_THRESHOLDS


def test_default_dpd_values() -> None:
    spec = DEFAULT_THRESHOLDS["demographic_parity_difference"]
    assert spec["fail"] == pytest.approx(0.10)
    assert spec["warn"] == pytest.approx(0.05)
    assert spec["direction"] == "lower_is_better"


def test_default_srr_values() -> None:
    spec = DEFAULT_THRESHOLDS["selection_rate_ratio"]
    assert spec["fail"] == pytest.approx(0.80)
    assert spec["warn"] == pytest.approx(0.90)
    assert spec["direction"] == "higher_is_better"


# ── resolve_thresholds ─────────────────────────────────────────────────────────

def test_resolve_no_override_returns_default() -> None:
    spec = resolve_thresholds("demographic_parity_difference", {})
    assert spec == DEFAULT_THRESHOLDS["demographic_parity_difference"]


def test_resolve_with_override_changes_fail_and_warn() -> None:
    spec = resolve_thresholds(
        "demographic_parity_difference",
        {"demographic_parity_difference": 0.20},
    )
    assert spec["fail"] == pytest.approx(0.20)
    assert spec["warn"] == pytest.approx(0.10)  # 0.20 / 2


def test_resolve_ratio_override_derives_warn_correctly() -> None:
    spec = resolve_thresholds(
        "selection_rate_ratio",
        {"selection_rate_ratio": 0.70},
    )
    assert spec["fail"] == pytest.approx(0.70)
    assert spec["warn"] == pytest.approx(0.85)  # (1.0 + 0.70) / 2


def test_resolve_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        resolve_thresholds("nonexistent_metric", {})


# ── evaluate_metric — lower_is_better (DPD, EOD) ──────────────────────────────

def test_lower_is_better_pass() -> None:
    assert evaluate_metric("demographic_parity_difference", 0.03, {}) == "pass"


def test_lower_is_better_at_warn_boundary() -> None:
    # exactly at warn=0.05: must be > warn to trigger warn, so 0.05 → pass
    assert evaluate_metric("demographic_parity_difference", 0.05, {}) == "pass"


def test_lower_is_better_warn() -> None:
    assert evaluate_metric("demographic_parity_difference", 0.07, {}) == "warn"


def test_lower_is_better_at_fail_boundary() -> None:
    # exactly at fail=0.10: must be > fail to trigger fail, so 0.10 → warn
    assert evaluate_metric("demographic_parity_difference", 0.10, {}) == "warn"


def test_lower_is_better_fail() -> None:
    assert evaluate_metric("demographic_parity_difference", 0.15, {}) == "fail"


# ── evaluate_metric — higher_is_better (SRR) ──────────────────────────────────

def test_higher_is_better_pass() -> None:
    assert evaluate_metric("selection_rate_ratio", 0.95, {}) == "pass"


def test_higher_is_better_at_warn_boundary() -> None:
    # exactly at warn=0.90: must be < warn to trigger warn, so 0.90 → pass
    assert evaluate_metric("selection_rate_ratio", 0.90, {}) == "pass"


def test_higher_is_better_warn() -> None:
    assert evaluate_metric("selection_rate_ratio", 0.85, {}) == "warn"


def test_higher_is_better_at_fail_boundary() -> None:
    # exactly at fail=0.80: must be < fail to trigger fail, so 0.80 → warn
    assert evaluate_metric("selection_rate_ratio", 0.80, {}) == "warn"


def test_higher_is_better_fail() -> None:
    assert evaluate_metric("selection_rate_ratio", 0.72, {}) == "fail"


# ── evaluate_metric — user override ───────────────────────────────────────────

def test_user_override_relaxes_fail_boundary() -> None:
    # override fail to 0.20; value 0.12 should now warn (between new warn=0.10 and fail=0.20)
    result = evaluate_metric(
        "demographic_parity_difference",
        0.12,
        {"demographic_parity_difference": 0.20},
    )
    assert result == "warn"


def test_user_override_value_below_new_warn_passes() -> None:
    # override fail to 0.20 → warn=0.10; value 0.08 should pass
    result = evaluate_metric(
        "demographic_parity_difference",
        0.08,
        {"demographic_parity_difference": 0.20},
    )
    assert result == "pass"


# ── compute_headline_score ─────────────────────────────────────────────────────

def test_headline_score_all_pass() -> None:
    assert compute_headline_score(["pass", "pass", "pass"]) == pytest.approx(1.0)


def test_headline_score_all_fail() -> None:
    assert compute_headline_score(["fail", "fail", "fail"]) == pytest.approx(0.0)


def test_headline_score_all_warn() -> None:
    assert compute_headline_score(["warn", "warn"]) == pytest.approx(0.5)


def test_headline_score_mixed() -> None:
    # pass=1.0, warn=0.5, fail=0.0 → mean = (1.0 + 0.5 + 0.0) / 3
    assert compute_headline_score(["pass", "warn", "fail"]) == pytest.approx(0.5)


def test_headline_score_empty_returns_zero() -> None:
    assert compute_headline_score([]) == pytest.approx(0.0)


# ── derive_headline_label ──────────────────────────────────────────────────────

def test_label_all_pass_is_good() -> None:
    assert derive_headline_label(["pass", "pass"]) == "good"


def test_label_any_fail_is_poor() -> None:
    assert derive_headline_label(["pass", "warn", "fail"]) == "poor"


def test_label_warn_only_is_moderate() -> None:
    assert derive_headline_label(["pass", "warn", "pass"]) == "moderate"


def test_label_single_fail_is_poor() -> None:
    assert derive_headline_label(["fail"]) == "poor"
