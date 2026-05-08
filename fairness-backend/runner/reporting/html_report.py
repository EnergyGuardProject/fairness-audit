"""Self-contained HTML report renderer using Jinja2 + Plotly.

Renders report.html.j2 with Plotly figures embedded as HTML fragments
(include_plotlyjs=True on the primary chart; CDN-free, fully offline).

Three chart slots per REPORT_SCHEMA.md:
    primary_chart   — bar chart: base metric (accuracy) per group
    secondary_chart — bar chart: selection rate per group
    tertiary_chart  — heatmap: disparity across sensitive feature combinations
                      (None if < 2 sensitive features)

When ≥ 3 sensitive features are present and intersectional=True, the two
features with the highest range of accuracy values across groups are selected
for the heatmap. A HEATMAP_FEATURE_SELECTION info-level warning is emitted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from runner.config.models import RunConfig
from runner.engine import FairnessResult

logger = logging.getLogger(__name__)


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
    raise NotImplementedError


def _build_primary_chart(
    result: FairnessResult,
    config: RunConfig,
) -> str:
    """Build the primary bar chart (accuracy per group) as an HTML fragment.

    Args:
        result: FairnessResult containing by_group accuracy values.
        config: RunConfig (provides base_metric label).

    Returns:
        Plotly figure HTML string with include_plotlyjs=True.
    """
    raise NotImplementedError


def _build_secondary_chart(result: FairnessResult) -> str:
    """Build the secondary bar chart (selection rate per group) as HTML.

    Args:
        result: FairnessResult containing by_group selection_rate values.

    Returns:
        Plotly figure HTML string (include_plotlyjs=False — already included).
    """
    raise NotImplementedError


def _build_tertiary_chart(
    result: FairnessResult,
    config: RunConfig,
) -> str | None:
    """Build the heatmap chart for multi-feature disparity, or return None.

    Returns None if < 2 sensitive features are configured.
    When ≥ 3 features and intersectional=True, selects the two features with
    the highest range of accuracy across groups and emits a
    HEATMAP_FEATURE_SELECTION info warning.

    Args:
        result: FairnessResult containing by_group metrics.
        config: RunConfig providing sensitive_features list.

    Returns:
        Plotly heatmap HTML string, or None.
    """
    raise NotImplementedError
