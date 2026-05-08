"""Pydantic models mirroring docs/REPORT_SCHEMA.md v1.0 for smoke test validation.

Usage in tests:
    import json
    from tests.fixtures.report_schema_validator import ReportPayload

    data = json.loads(metrics_user_path.read_text())
    payload = ReportPayload.model_validate(data)  # raises ValidationError on schema mismatch
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ReportMeta(BaseModel):
    """report_meta block — identical across all T5.2 services."""

    service: str
    service_version: str
    run_id: str
    run_name: str
    description: str | None
    task_type: Literal["classification", "regression"]
    model_backend: Literal["sklearn", "pytorch", "mlflow"]
    mlflow_run_id: str | None
    mlflow_tracking_uri: str | None
    dataset_uri: str
    feature_count: int
    sample_count: int
    samples_evaluated: int
    timestamp_utc: str
    status: Literal["ok", "partial", "error"]


class PrimaryMetric(BaseModel):
    """A single entry in summary.primary_metrics."""

    key: str
    label: str
    value: float
    format: Literal["scalar", "percent", "ratio"]
    direction: Literal["lower_is_better", "higher_is_better"]
    threshold: float
    status: Literal["pass", "warn", "fail"]


class Summary(BaseModel):
    """summary block."""

    headline_score: float
    headline_label: Literal["good", "moderate", "poor"]
    primary_metrics: list[PrimaryMetric]


class SubgroupRow(BaseModel):
    """A single row in subgroup_breakdown."""

    row_label: str
    row_kind: str
    metrics: dict[str, float | None]
    n_samples: int
    status: Literal["pass", "warn", "fail"]


class ChartSpec(BaseModel):
    """A single chart slot."""

    type: Literal["bar", "line", "scatter", "heatmap"]
    title: str
    x_label: str
    y_label: str
    points: list[Any]
    reference_value: float | None = None
    annotations: list[Any] = []


class Charts(BaseModel):
    """charts block — three named slots."""

    primary_chart: ChartSpec
    secondary_chart: ChartSpec | None
    tertiary_chart: ChartSpec | None


class Warning(BaseModel):
    """A single warning entry."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class Recommendation(BaseModel):
    """A single recommendation entry."""

    priority: Literal["high", "medium", "low"]
    category: Literal["data", "model", "deployment", "documentation"]
    action: str
    rationale: str
    external_refs: list[str]


class ReportPayload(BaseModel):
    """Top-level metrics_user.json structure per REPORT_SCHEMA.md v1.0."""

    schema_version: str
    report_meta: ReportMeta
    evaluation_setup: dict[str, Any]
    summary: Summary
    subgroup_breakdown: list[SubgroupRow]
    charts: Charts
    warnings: list[Warning]
    recommendations: list[Recommendation]
