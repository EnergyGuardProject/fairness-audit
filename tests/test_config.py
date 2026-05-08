"""Tests for RunConfig and related Pydantic config models.

These tests exercise runner/config/models.py directly, with no heavy
dependencies (no fairlearn, no sklearn, no pandas).
"""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from runner.config.models import (
    DatasetConfig,
    FairnessConfig,
    FairnessMetric,
    ModelConfig,
    RunConfig,
)

_VALID_YAML = """
run_name: "Test Run"
dataset:
  path: /tmp/test.csv
  target_column: label
model:
  path: /tmp/model.joblib
fairness:
  sensitive_features:
    - sex
    - race
"""


def _parse(yaml_str: str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(yaml_str))


# ── Happy-path ─────────────────────────────────────────────────────────────────

def test_valid_config_parses() -> None:
    config = _parse(_VALID_YAML)
    assert config.run_name == "Test Run"
    assert config.description is None
    assert config.fairness.sensitive_features == ["sex", "race"]
    assert config.fairness.intersectional is False
    assert config.model.backend == "sklearn"
    assert config.dataset.format == "csv"
    assert len(config.fairness.fairness_metrics) == 3


def test_default_fairness_metrics() -> None:
    config = _parse(_VALID_YAML)
    keys = [m.value for m in config.fairness.fairness_metrics]
    assert "demographic_parity_difference" in keys
    assert "equalized_odds_difference" in keys
    assert "selection_rate_ratio" in keys


def test_description_optional() -> None:
    config = _parse(_VALID_YAML)
    assert config.description is None

    with_desc = _VALID_YAML + 'description: "My audit"\n'
    config2 = _parse(with_desc)
    assert config2.description == "My audit"


def test_intersectional_defaults_false() -> None:
    config = _parse(_VALID_YAML)
    assert config.fairness.intersectional is False


def test_parity_threshold_override_preserved() -> None:
    yaml_str = _VALID_YAML + (
        "  parity_thresholds:\n"
        "    demographic_parity_difference: 0.15\n"
    )
    config = _parse(yaml_str)
    assert config.fairness.parity_thresholds["demographic_parity_difference"] == pytest.approx(0.15)


def test_run_name_stripped() -> None:
    yaml_str = _VALID_YAML.replace('"Test Run"', '"  padded  "')
    config = _parse(yaml_str)
    assert config.run_name == "padded"


# ── Validation errors ──────────────────────────────────────────────────────────

def test_empty_sensitive_features_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["fairness"]["sensitive_features"] = []
    with pytest.raises(ValidationError, match="sensitive_features must be non-empty"):
        RunConfig.model_validate(data)


def test_duplicate_sensitive_features_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["fairness"]["sensitive_features"] = ["sex", "sex"]
    with pytest.raises(ValidationError, match="duplicates"):
        RunConfig.model_validate(data)


def test_invalid_backend_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["model"]["backend"] = "pytorch"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_invalid_format_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["dataset"]["format"] = "parquet"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_parity_threshold_invalid_key_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["fairness"]["parity_thresholds"] = {"nonexistent_metric": 0.1}
    with pytest.raises(ValidationError, match="not a valid FairnessMetric"):
        RunConfig.model_validate(data)


def test_parity_threshold_nonpositive_value_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["fairness"]["parity_thresholds"] = {"demographic_parity_difference": 0.0}
    with pytest.raises(ValidationError, match="must be > 0"):
        RunConfig.model_validate(data)


def test_blank_run_name_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    data["run_name"] = "   "
    with pytest.raises(ValidationError, match="must not be blank"):
        RunConfig.model_validate(data)


def test_missing_target_column_raises() -> None:
    data = yaml.safe_load(_VALID_YAML)
    del data["dataset"]["target_column"]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)
