"""End-to-end smoke tests for the fairness audit pipeline.

These tests load the committed example model + dataset, run the full pipeline
via the CLI entry point, and assert specific metric statuses.

Prerequisites (must run first):
    python examples/scripts/generate_synthetic_energy_burden.py
    python examples/scripts/train_baseline_model.py

The tests are skipped automatically if the example data files are absent,
with a clear message indicating which generation step is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

EXAMPLE_MODEL = Path("examples/models/baseline_logreg.joblib")
EXAMPLE_DATA = Path("examples/data/energy_burden_synthetic.csv")
EXAMPLE_CONFIG = Path("examples/configs/energy_burden.yaml")

_missing_model = not EXAMPLE_MODEL.exists()
_missing_data = not EXAMPLE_DATA.exists()

_skip_reason = " ".join(filter(None, [
    "Missing example files:" if (_missing_model or _missing_data) else "",
    str(EXAMPLE_MODEL) if _missing_model else "",
    str(EXAMPLE_DATA) if _missing_data else "",
    "(run generate_synthetic_energy_burden.py and train_baseline_model.py)" if (
        _missing_model or _missing_data
    ) else "",
]))

pytestmark = pytest.mark.skipif(
    _missing_model or _missing_data,
    reason=_skip_reason or "example files present",
)


@pytest.fixture(scope="module")
def audit_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full CLI pipeline once; return the job output directory."""
    import subprocess
    import sys

    output_root = tmp_path_factory.mktemp("runs")
    result = subprocess.run(
        [
            sys.executable, "-m", "runner", "run",
            "--config", str(EXAMPLE_CONFIG),
            "--output", str(output_root),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI exited with code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    job_id = result.stdout.strip()
    assert job_id, "CLI produced no job_id on stdout"
    return output_root / job_id


def test_artifacts_exist(audit_output: Path) -> None:
    """All expected output files are present."""
    assert (audit_output / "metrics.json").exists()
    assert (audit_output / "metrics_user.json").exists()
    assert (audit_output / "report.html").exists()
    assert (audit_output / "config_resolved.yaml").exists()
    assert (audit_output / "run.log").exists()


def test_metrics_json_is_valid_json(audit_output: Path) -> None:
    data = json.loads((audit_output / "metrics.json").read_text())
    assert "provenance" in data
    assert "primary_metrics" in data


def test_metrics_user_validates_against_schema(audit_output: Path) -> None:
    """metrics_user.json must parse cleanly against the ReportPayload schema."""
    from pydantic import ValidationError
    from tests.fixtures.report_schema_validator import ReportPayload

    data = json.loads((audit_output / "metrics_user.json").read_text())
    try:
        ReportPayload.model_validate(data)
    except ValidationError as exc:
        pytest.fail(f"metrics_user.json does not conform to REPORT_SCHEMA.md:\n{exc}")


def test_equalized_odds_difference_fails(audit_output: Path) -> None:
    """EOD must be 'fail' — disparity is deliberately injected into the dataset."""
    data = json.loads((audit_output / "metrics_user.json").read_text())
    metrics = {m["key"]: m for m in data["summary"]["primary_metrics"]}
    assert "equalized_odds_difference" in metrics, "EOD not in primary_metrics"
    assert metrics["equalized_odds_difference"]["status"] == "fail", (
        f"Expected EOD status 'fail', got {metrics['equalized_odds_difference']['status']!r}. "
        f"EOD value: {metrics['equalized_odds_difference']['value']:.4f}"
    )


def test_selection_rate_ratio_fails(audit_output: Path) -> None:
    """SRR between best and worst income decile must be 'fail'."""
    data = json.loads((audit_output / "metrics_user.json").read_text())
    metrics = {m["key"]: m for m in data["summary"]["primary_metrics"]}
    assert "selection_rate_ratio" in metrics, "SRR not in primary_metrics"
    assert metrics["selection_rate_ratio"]["status"] == "fail", (
        f"Expected SRR status 'fail', got {metrics['selection_rate_ratio']['status']!r}. "
        f"SRR value: {metrics['selection_rate_ratio']['value']:.4f}"
    )


def test_headline_label_is_poor(audit_output: Path) -> None:
    data = json.loads((audit_output / "metrics_user.json").read_text())
    assert data["summary"]["headline_label"] == "poor"


def test_subgroup_breakdown_has_income_decile_rows(audit_output: Path) -> None:
    """At least 10 subgroup rows exist (one per income_decile value)."""
    data = json.loads((audit_output / "metrics_user.json").read_text())
    assert len(data["subgroup_breakdown"]) >= 10


def test_report_html_is_self_contained(audit_output: Path) -> None:
    """HTML report must include Plotly JS (self-contained, offline-usable)."""
    html = (audit_output / "report.html").read_text()
    assert "plotly" in html.lower()
    assert len(html) > 1000  # not an empty stub


def test_smoke_is_deterministic(tmp_path: Path) -> None:
    """Two runs with the same config produce identical primary metric values."""
    import subprocess
    import sys

    runs = []
    for _ in range(2):
        out = tmp_path / f"run_{_}"
        result = subprocess.run(
            [
                sys.executable, "-m", "runner", "run",
                "--config", str(EXAMPLE_CONFIG),
                "--output", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        job_id = result.stdout.strip()
        payload = json.loads((out / job_id / "metrics_user.json").read_text())
        runs.append(payload["summary"]["primary_metrics"])

    for m1, m2 in zip(runs[0], runs[1]):
        assert m1["key"] == m2["key"]
        assert m1["value"] == pytest.approx(m2["value"], abs=1e-9), (
            f"Non-deterministic result for {m1['key']}: {m1['value']} vs {m2['value']}"
        )
