"""Typer CLI for the EnergyGuard Fairness Audit runner.

Usage:
    python -m runner run --config <path>
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="runner",
    help="EnergyGuard Fairness Audit CLI.",
    add_completion=False,
)

logger = logging.getLogger(__name__)


@app.callback()
def _callback() -> None:
    """EnergyGuard Fairness Audit CLI."""


@app.command("run")
def run_command(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the YAML RunConfig file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("runs"),
        "--output",
        "-o",
        help="Root directory for job output. Job artifacts go into <output>/<job_id>/.",
    ),
    job_id: str | None = typer.Option(
        None,
        "--job-id",
        help="Override the generated job_id (useful for reproducibility checks).",
    ),
) -> None:
    """Run a fairness audit end-to-end from a YAML config file.

    Writes the following artifacts to <output>/<job_id>/:
        config_resolved.yaml  — resolved config with absolute paths
        run.log               — structured log for this job
        metrics.json          — firehose metrics + provenance block
        metrics_user.json     — schema-conforming report payload
        report.html           — self-contained Plotly report

    Exits with code 0 on success, 1 on error.

    Args:
        config: Path to YAML RunConfig.
        output: Output root directory.
        job_id: Optional override for the generated job_id.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from runner.pipeline import run as pipeline_run

    try:
        resolved_job_id = pipeline_run(
            config_path=config,
            output_root=output,
            job_id_override=job_id,
        )
        typer.echo(resolved_job_id)
    except Exception as exc:
        logger.error("Audit failed: %s", exc, exc_info=True)
        typer.echo(f"Error: {exc}", err=True)
        sys.exit(1)
