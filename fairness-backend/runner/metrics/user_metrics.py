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
from runner.metrics.thresholds import DEFAULT_THRESHOLDS, evaluate_metric, resolve_thresholds

logger = logging.getLogger(__name__)

_STATUS_SCORE: dict[str, float] = {"pass": 1.0, "warn": 0.5, "fail": 0.0}

# Metrics that appear in summary.primary_metrics with pass/warn/fail status.
THRESHOLDED_METRICS = frozenset({
    "demographic_parity_difference",
    "equalized_odds_difference",
    "selection_rate_ratio",
})

_METRIC_LABELS: dict[str, str] = {
    "demographic_parity_difference": "Demographic Parity Difference",
    "equalized_odds_difference": "Equalized Odds Difference",
    "selection_rate_ratio": "Selection Rate Ratio",
}

_METRIC_FORMATS: dict[str, str] = {
    "demographic_parity_difference": "scalar",
    "equalized_odds_difference": "scalar",
    "selection_rate_ratio": "ratio",
}

# Code-to-severity mapping; fallback is "warning".
_WARNING_SEVERITY: dict[str, str] = {
    "ALL_NEGATIVE_PREDICTIONS": "error",
    "INSUFFICIENT_GROUPS_FOR_METRIC": "warning",
}

_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "demographic_parity_difference": {
        "priority": "high",
        "category": "model",
        "action": (
            "Investigate selection rate disparities across groups and consider "
            "reweighing or threshold optimisation to reduce the demographic parity gap."
        ),
        "rationale": (
            "Demographic Parity Difference exceeds the configured threshold, "
            "indicating unequal positive prediction rates across protected groups."
        ),
        "external_refs": [
            "https://fairlearn.org/main/user_guide/mitigation/reweighing.html"
        ],
    },
    "equalized_odds_difference": {
        "priority": "high",
        "category": "model",
        "action": (
            "Apply equalized odds post-processing or use an in-processing method "
            "that constrains FPR and FNR gaps across groups."
        ),
        "rationale": (
            "Equalized Odds Difference exceeds threshold, indicating systematic "
            "differences in false positive or false negative rates across groups."
        ),
        "external_refs": [
            "https://fairlearn.org/main/user_guide/mitigation/postprocessing.html"
        ],
    },
    "selection_rate_ratio": {
        "priority": "high",
        "category": "data",
        "action": (
            "Review label distribution and feature representation by group; "
            "consider data augmentation or resampling to balance positive prediction rates."
        ),
        "rationale": (
            "Selection Rate Ratio is below threshold, indicating the lowest-represented "
            "group receives positive predictions far less often than the most-represented group."
        ),
        "external_refs": [
            "https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html"
        ],
    },
}


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
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the full metrics_user.json payload from a FairnessResult.

    Args:
        result: Output of FairnessEngine.run().
        config: The RunConfig used for this evaluation.
        provenance: Dict containing job_id, timestamp_utc, service_version,
            and feature_count (plus library versions and config_hash).

    Returns:
        Dict conforming to docs/REPORT_SCHEMA.md v1.0.
    """
    job_id: str = provenance["job_id"]
    timestamp_utc: str = provenance["timestamp_utc"]
    service_version: str = provenance["service_version"]
    feature_count: int = provenance["feature_count"]
    parity_thresholds: dict[str, float] = config.fairness.parity_thresholds

    extra_warnings: list[dict[str, str]] = []

    # ── summary.primary_metrics ─────────────────────────────────────────────
    primary_metrics: list[dict[str, Any]] = []
    statuses: list[Literal["pass", "warn", "fail"]] = []

    for pmv in result.primary_metrics:
        if pmv.key not in THRESHOLDED_METRICS:
            continue
        if pmv.value is None:
            extra_warnings.append({
                "code": "METRIC_NOT_COMPUTABLE",
                "severity": "warning",
                "message": f"Could not compute {pmv.key}: insufficient valid groups.",
            })
            continue

        spec = resolve_thresholds(pmv.key, parity_thresholds)
        status = evaluate_metric(pmv.key, pmv.value, parity_thresholds)
        statuses.append(status)

        primary_metrics.append({
            "key": pmv.key,
            "label": _METRIC_LABELS.get(pmv.key, pmv.key),
            "value": round(pmv.value, 6),
            "format": _METRIC_FORMATS.get(pmv.key, "scalar"),
            "direction": spec["direction"],
            "threshold": spec["fail"],
            "status": status,
        })

    headline_score = compute_headline_score(statuses)
    headline_label = derive_headline_label(statuses)

    # ── evaluation_setup ─────────────────────────────────────────────────────
    effective_thresholds: dict[str, float] = {}
    for mkey in [m.value for m in config.fairness.fairness_metrics]:
        if mkey in DEFAULT_THRESHOLDS:
            effective_thresholds[mkey] = resolve_thresholds(mkey, parity_thresholds)["fail"]

    evaluation_setup: dict[str, Any] = {
        "sensitive_features": config.fairness.sensitive_features,
        "fairness_metrics": [m.value for m in config.fairness.fairness_metrics],
        "parity_thresholds": effective_thresholds,
        "base_metric": config.fairness.base_metric,
    }

    # ── subgroup_breakdown ───────────────────────────────────────────────────
    subgroup_breakdown: list[dict[str, Any]] = []
    for group_key in sorted(result.by_group.keys()):
        gm = result.by_group[group_key]
        gap = abs(gm.accuracy - result.overall_accuracy)
        if gap > 0.10:
            g_status: str = "fail"
        elif gap > 0.05:
            g_status = "warn"
        else:
            g_status = "pass"

        fpr = round(gm.false_positive_rate, 6) if gm.false_positive_rate is not None else None
        fnr = round(gm.false_negative_rate, 6) if gm.false_negative_rate is not None else None

        subgroup_breakdown.append({
            "row_label": group_key,
            "row_kind": "group",
            "metrics": {
                "accuracy": round(gm.accuracy, 6),
                "selection_rate": round(gm.selection_rate, 6),
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
            },
            "n_samples": gm.n_samples,
            "status": g_status,
        })

    # ── charts ───────────────────────────────────────────────────────────────
    sorted_groups = sorted(result.by_group.keys())

    primary_chart: dict[str, Any] = {
        "type": "bar",
        "title": f"{config.fairness.base_metric.capitalize()} by Group",
        "x_label": "group",
        "y_label": config.fairness.base_metric,
        "points": [
            [gk, round(result.by_group[gk].accuracy, 6)] for gk in sorted_groups
        ],
        "reference_value": round(result.overall_accuracy, 6),
        "annotations": [],
    }
    secondary_chart: dict[str, Any] = {
        "type": "bar",
        "title": "Selection Rate by Group",
        "x_label": "group",
        "y_label": "selection_rate",
        "points": [
            [gk, round(result.by_group[gk].selection_rate, 6)] for gk in sorted_groups
        ],
        "reference_value": round(result.overall_selection_rate, 6),
        "annotations": [],
    }

    charts: dict[str, Any] = {
        "primary_chart": primary_chart,
        "secondary_chart": secondary_chart,
        "tertiary_chart": None,
    }

    # ── warnings ─────────────────────────────────────────────────────────────
    warnings: list[dict[str, str]] = []
    for w in result.warnings:
        warnings.append({
            "code": w.code,
            "severity": _WARNING_SEVERITY.get(w.code, "warning"),
            "message": w.message,
        })
    warnings.extend(extra_warnings)

    # ── recommendations — one per failing primary metric ─────────────────────
    recommendations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for m in primary_metrics:
        mkey = m["key"]
        if m["status"] == "fail" and mkey in _RECOMMENDATIONS and mkey not in seen_keys:
            recommendations.append(_RECOMMENDATIONS[mkey])
            seen_keys.add(mkey)

    return {
        "schema_version": "1.0",
        "report_meta": {
            "service": "fairness",
            "service_version": service_version,
            "run_id": job_id,
            "run_name": config.run_name,
            "description": config.description,
            "task_type": "classification",
            "model_backend": config.model.backend,
            "mlflow_run_id": None,
            "mlflow_tracking_uri": None,
            "dataset_uri": config.dataset.path,
            "feature_count": feature_count,
            "sample_count": result.n_samples,
            "samples_evaluated": result.n_samples,
            "timestamp_utc": timestamp_utc,
            "status": "ok",
        },
        "evaluation_setup": evaluation_setup,
        "summary": {
            "headline_score": round(headline_score, 6),
            "headline_label": headline_label,
            "primary_metrics": primary_metrics,
        },
        "subgroup_breakdown": subgroup_breakdown,
        "charts": charts,
        "warnings": warnings,
        "recommendations": recommendations,
    }
