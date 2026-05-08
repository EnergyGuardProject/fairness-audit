"""Tests for the FastAPI service endpoints (app/main.py).

Fixtures:
    api_fixtures  — minimal .joblib model + CSV dataset + config YAML bytes
    api_client    — TestClient with RUNS_DIR/TMP_DIR monkeypatched to tmp dirs
    completed_job — POST once; shares the resulting job_id across module tests

Notes:
    - TestClient (Starlette) runs BackgroundTasks synchronously before returning,
      so by the time client.post() returns the background job is already complete.
    - 409 tests create job dirs manually to simulate a pending/running state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_fixtures(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, bytes]:
    """Build a minimal model, CSV dataset, and YAML config bytes.

    Model is trained on [feat1, group]; dataset target column is "target";
    sensitive feature is "group" (binary 0/1).
    """
    d = tmp_path_factory.mktemp("api_fixtures")

    rng = np.random.default_rng(42)
    n = 40
    group = rng.integers(0, 2, n)
    feat1 = rng.standard_normal(n) + group.astype(float) * 0.5
    target = (feat1 + group.astype(float) * 0.5 > 0).astype(int)

    df = pd.DataFrame({"feat1": feat1, "target": target, "group": group})
    csv_path = d / "dataset.csv"
    df.to_csv(csv_path, index=False)

    X = df[["feat1", "group"]].to_numpy()
    model = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    model.fit(X, target)
    model_path = d / "model.joblib"
    joblib.dump(model, model_path)

    config_bytes = (
        b"run_name: test-api-run\n"
        b"dataset:\n"
        b"  format: csv\n"
        b"  target_column: target\n"
        b"model:\n"
        b"  backend: sklearn\n"
        b"fairness:\n"
        b"  sensitive_features:\n"
        b"    - group\n"
    )

    return model_path, csv_path, config_bytes


@pytest.fixture(scope="module")
def api_client(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[tuple[TestClient, Path], None, None]:
    """TestClient with RUNS_DIR and TMP_DIR monkeypatched to isolated tmp dirs."""
    from app import main as main_module

    mp = pytest.MonkeyPatch()
    runs_dir = tmp_path_factory.mktemp("api_runs")
    upload_dir = tmp_path_factory.mktemp("api_uploads")
    runs_dir.mkdir(exist_ok=True)
    upload_dir.mkdir(exist_ok=True)
    mp.setattr(main_module, "RUNS_DIR", runs_dir)
    mp.setattr(main_module, "TMP_DIR", upload_dir)

    with TestClient(main_module.app) as client:
        yield client, runs_dir

    mp.undo()


@pytest.fixture(scope="module")
def completed_job(
    api_client: tuple[TestClient, Path],
    api_fixtures: tuple[Path, Path, bytes],
) -> tuple[TestClient, Path, str]:
    """POST a valid evaluation once; return (client, runs_dir, job_id).

    BackgroundTasks run synchronously in TestClient, so the job is fully
    complete (status == "ok") before this fixture returns.
    """
    client, runs_dir = api_client
    model_path, csv_path, config_bytes = api_fixtures

    response = client.post(
        "/api/evaluations",
        files={
            "model_file": ("model.joblib", model_path.read_bytes(), "application/octet-stream"),
            "dataset_file": ("dataset.csv", csv_path.read_bytes(), "text/csv"),
            "config": ("config.yaml", config_bytes, "application/x-yaml"),
        },
    )
    assert response.status_code == 202, f"POST failed:\n{response.text}"
    job_id = response.json()["job_id"]
    return client, runs_dir, job_id


def _post_evaluation(
    client: TestClient,
    model_path: Path,
    csv_path: Path,
    config_bytes: bytes,
) -> "requests.Response":  # type: ignore[name-defined]
    return client.post(
        "/api/evaluations",
        files={
            "model_file": ("model.joblib", model_path.read_bytes(), "application/octet-stream"),
            "dataset_file": ("dataset.csv", csv_path.read_bytes(), "text/csv"),
            "config": ("config.yaml", config_bytes, "application/x-yaml"),
        },
    )


def _make_pending_job(runs_dir: Path, job_id: str) -> None:
    """Write a minimal pending status.json to simulate a pre-completion job."""
    job_dir = runs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(
        json.dumps({"status": "pending", "created_at": "2025-01-01T00:00:00Z"}),
        encoding="utf-8",
    )


# ── POST /api/evaluations ──────────────────────────────────────────────────────

def test_post_returns_202_with_pending_status(
    api_client: tuple[TestClient, Path],
    api_fixtures: tuple[Path, Path, bytes],
) -> None:
    client, _ = api_client
    model_path, csv_path, config_bytes = api_fixtures

    response = _post_evaluation(client, model_path, csv_path, config_bytes)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body.get("job_id")


def test_post_invalid_yaml_returns_400(
    api_client: tuple[TestClient, Path],
    api_fixtures: tuple[Path, Path, bytes],
) -> None:
    client, _ = api_client
    model_path, csv_path, _ = api_fixtures

    bad_yaml = b"run_name: test\nfairness: {unclosed: [bracket\n"
    response = _post_evaluation(client, model_path, csv_path, bad_yaml)

    assert response.status_code == 400


def test_post_invalid_config_returns_422(
    api_client: tuple[TestClient, Path],
    api_fixtures: tuple[Path, Path, bytes],
) -> None:
    client, _ = api_client
    model_path, csv_path, _ = api_fixtures

    # sensitive_features: [] violates the non-empty validator
    bad_config = (
        b"run_name: test\n"
        b"dataset:\n"
        b"  format: csv\n"
        b"  target_column: target\n"
        b"model:\n"
        b"  backend: sklearn\n"
        b"fairness:\n"
        b"  sensitive_features: []\n"
    )
    response = _post_evaluation(client, model_path, csv_path, bad_config)

    assert response.status_code == 422


# ── GET /api/evaluations/{job_id}/status ──────────────────────────────────────

def test_get_status_ok_after_completed_job(
    completed_job: tuple[TestClient, Path, str],
) -> None:
    client, _, job_id = completed_job

    response = client.get(f"/api/evaluations/{job_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "ok"
    assert body["created_at"]
    assert body["updated_at"]


def test_get_status_unknown_job_returns_404(
    api_client: tuple[TestClient, Path],
) -> None:
    client, _ = api_client

    response = client.get("/api/evaluations/nonexistent-xyz-000/status")

    assert response.status_code == 404


def test_get_status_pending_job_returns_pending(
    api_client: tuple[TestClient, Path],
) -> None:
    client, runs_dir = api_client
    job_id = "manual-pending-status"
    _make_pending_job(runs_dir, job_id)

    response = client.get(f"/api/evaluations/{job_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    # created_at has no completed_at/updated_at → updated_at falls back to created_at
    assert body["updated_at"] == "2025-01-01T00:00:00Z"


# ── GET /api/evaluations/{job_id}/metrics ─────────────────────────────────────

def test_get_metrics_ok(
    completed_job: tuple[TestClient, Path, str],
) -> None:
    client, _, job_id = completed_job

    response = client.get(f"/api/evaluations/{job_id}/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "subgroup_breakdown" in body


def test_get_metrics_unknown_returns_404(
    api_client: tuple[TestClient, Path],
) -> None:
    client, _ = api_client

    response = client.get("/api/evaluations/nonexistent-xyz-001/metrics")

    assert response.status_code == 404


def test_get_metrics_pending_returns_409(
    api_client: tuple[TestClient, Path],
) -> None:
    client, runs_dir = api_client
    job_id = "manual-pending-metrics"
    _make_pending_job(runs_dir, job_id)

    response = client.get(f"/api/evaluations/{job_id}/metrics")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["job_id"] == job_id
    assert detail["status"] == "pending"


# ── GET /api/evaluations/{job_id}/report ──────────────────────────────────────

def test_get_report_ok(
    completed_job: tuple[TestClient, Path, str],
) -> None:
    client, _, job_id = completed_job

    response = client.get(f"/api/evaluations/{job_id}/report")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "plotly" in response.text.lower()


def test_get_report_unknown_returns_404(
    api_client: tuple[TestClient, Path],
) -> None:
    client, _ = api_client

    response = client.get("/api/evaluations/nonexistent-xyz-002/report")

    assert response.status_code == 404


def test_get_report_pending_returns_409(
    api_client: tuple[TestClient, Path],
) -> None:
    client, runs_dir = api_client
    job_id = "manual-pending-report"
    _make_pending_job(runs_dir, job_id)

    response = client.get(f"/api/evaluations/{job_id}/report")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["job_id"] == job_id
    assert detail["status"] == "pending"
