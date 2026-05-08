"""Self-contained HTML report renderer using Jinja2 + Plotly.

Renders report.html.j2 with Plotly figures embedded as HTML fragments
(include_plotlyjs=True on the primary chart; CDN-free, fully offline).

Three chart slots per REPORT_SCHEMA.md:
    primary_chart   — bar chart: base metric (accuracy) per group
    secondary_chart — bar chart: selection rate per group
    tertiary_chart  — heatmap: deferred to v1.1; always None in v1

When ≥ 3 sensitive features are present and intersectional=True, the two
features with the highest range of accuracy values across groups are selected
for the heatmap. A HEATMAP_FEATURE_SELECTION info-level warning is emitted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import plotly.graph_objects as go  # type: ignore[import-untyped]
from jinja2 import Environment, FileSystemLoader

from runner.config.models import RunConfig
from runner.engine import FairnessResult

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _ascii_safe(s: str) -> str:
    """Replace non-ASCII Unicode characters with \\uXXXX JavaScript escapes.

    Plotly's embedded JS contains a handful of non-ASCII characters (µ and CJK
    chars inside regex literals). On Windows, pathlib.Path.read_text() uses the
    system locale codec (cp1252) which cannot decode these bytes. Replacing them
    with \\uXXXX is valid everywhere in JavaScript (regex character classes,
    string literals, identifiers) and produces a pure-ASCII file readable with
    any single-byte encoding.

    Only called on the primary chart fragment that carries include_plotlyjs=True.
    """
    return "".join(
        c if ord(c) < 128 else f"\\u{ord(c):04x}"
        for c in s
    )


def render_html_report(
    result: FairnessResult,
    metrics_user: dict[str, Any],
    config: RunConfig,
    output_path: Path,
) -> None:
    """Render the self-contained HTML report and write it to output_path.

    Args:
        result: FairnessResult from the engine (used for chart data).
        metrics_user: The metrics_user.json payload (provides schema-conforming
            summary, subgroup_breakdown, warnings, and recommendations).
        config: RunConfig for this evaluation.
        output_path: Destination file path for report.html.

    Raises:
        OSError: If output_path cannot be written.
    """
    primary_chart_html = _build_primary_chart(result, config)
    secondary_chart_html = _build_secondary_chart(result)
    tertiary_chart_html = _build_tertiary_chart(result, config)

    report_meta = metrics_user.get("report_meta", {})
    summary = metrics_user.get("summary", {})
    charts = metrics_user.get("charts", {})
    primary_chart_spec = charts.get("primary_chart") or {}
    secondary_chart_spec = charts.get("secondary_chart") or {}

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        run_name=report_meta.get("run_name", ""),
        timestamp_utc=report_meta.get("timestamp_utc", ""),
        service_version=report_meta.get("service_version", ""),
        dataset_uri=report_meta.get("dataset_uri", ""),
        sample_count=report_meta.get("sample_count", 0),
        model_backend=report_meta.get("model_backend", ""),
        headline_label=summary.get("headline_label", "poor"),
        headline_score=summary.get("headline_score", 0.0),
        headline_gauge_html=None,
        primary_metrics=summary.get("primary_metrics", []),
        primary_chart_html=primary_chart_html,
        primary_chart_title=primary_chart_spec.get("title", "Accuracy by Group"),
        secondary_chart_html=secondary_chart_html,
        secondary_chart_title=secondary_chart_spec.get("title", "Selection Rate by Group"),
        tertiary_chart_html=tertiary_chart_html,
        tertiary_chart_title=None,
        subgroup_breakdown=metrics_user.get("subgroup_breakdown", []),
        warnings=metrics_user.get("warnings", []),
        recommendations=metrics_user.get("recommendations", []),
        run_id=report_meta.get("run_id", ""),
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Written report.html to %s (%d bytes)", output_path, len(html))


def _build_primary_chart(
    result: FairnessResult,
    config: RunConfig,
) -> str:
    """Build the primary bar chart (accuracy per group) as an HTML fragment.

    Groups are sorted by key for determinism. Reference line at overall accuracy.

    Args:
        result: FairnessResult containing by_group accuracy values.
        config: RunConfig (provides base_metric label).

    Returns:
        Plotly figure HTML string with include_plotlyjs=True.
    """
    sorted_groups = sorted(result.by_group.keys())
    x_vals = sorted_groups
    y_vals = [round(result.by_group[g].accuracy, 6) for g in sorted_groups]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        name=config.fairness.base_metric,
        marker_color="#3b82f6",
    ))
    fig.add_hline(
        y=round(result.overall_accuracy, 6),
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Overall: {result.overall_accuracy:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"{config.fairness.base_metric.capitalize()} by Group",
        xaxis_title="Group",
        yaxis_title=config.fairness.base_metric.capitalize(),
        yaxis={"range": [0, 1]},
        xaxis={"tickangle": -45},
        margin={"l": 50, "r": 20, "t": 50, "b": 120},
        height=420,
    )
    return _ascii_safe(fig.to_html(include_plotlyjs=True, full_html=False))


def _build_secondary_chart(result: FairnessResult) -> str:
    """Build the secondary bar chart (selection rate per group) as HTML.

    Groups are sorted by key for determinism. Reference line at overall
    selection rate. include_plotlyjs=False because the primary chart already
    embedded the Plotly library.

    Args:
        result: FairnessResult containing by_group selection_rate values.

    Returns:
        Plotly figure HTML string (include_plotlyjs=False).
    """
    sorted_groups = sorted(result.by_group.keys())
    x_vals = sorted_groups
    y_vals = [round(result.by_group[g].selection_rate, 6) for g in sorted_groups]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals,
        y=y_vals,
        name="selection_rate",
        marker_color="#10b981",
    ))
    fig.add_hline(
        y=round(result.overall_selection_rate, 6),
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Overall: {result.overall_selection_rate:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Selection Rate by Group",
        xaxis_title="Group",
        yaxis_title="Selection Rate",
        yaxis={"range": [0, 1]},
        xaxis={"tickangle": -45},
        margin={"l": 50, "r": 20, "t": 50, "b": 120},
        height=420,
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _build_tertiary_chart(
    result: FairnessResult,
    config: RunConfig,
) -> str | None:
    """Build the heatmap chart for multi-feature disparity, or return None.

    Heatmap is deferred to v1.1. Always returns None in v1.

    Args:
        result: FairnessResult containing by_group metrics.
        config: RunConfig providing sensitive_features list.

    Returns:
        None in v1.
    """
    return None
