"""Pydantic v2 configuration models for the fairness audit runner.

All config is loaded from a YAML file via:
    import yaml
    from runner.config.models import RunConfig
    config = RunConfig.model_validate(yaml.safe_load(path.read_text()))
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Extend this set in v1.1 when precision/recall support lands.
# Using a frozenset + validator instead of Literal["accuracy"] avoids a
# breaking schema change when new base metrics are added.
_ALLOWED_BASE_METRICS: frozenset[str] = frozenset({"accuracy"})


class FairnessMetric(StrEnum):
    """Supported fairness metrics.

    Primary metrics (thresholded — appear in summary.primary_metrics):
        demographic_parity_difference, equalized_odds_difference,
        selection_rate_ratio

    Diagnostic metrics (not thresholded — appear in subgroup_breakdown only):
        false_positive_rate_difference, false_negative_rate_difference
    """

    demographic_parity_difference = "demographic_parity_difference"
    equalized_odds_difference = "equalized_odds_difference"
    selection_rate_ratio = "selection_rate_ratio"
    false_positive_rate_difference = "false_positive_rate_difference"
    false_negative_rate_difference = "false_negative_rate_difference"


class DatasetConfig(BaseModel):
    """Configuration for the input CSV dataset."""

    path: str = Field(..., description="Path to the CSV dataset file.")
    format: Literal["csv"] = Field("csv", description="Dataset format. Only 'csv' in v1.")
    target_column: str = Field(..., description="Name of the binary label column.")


class ModelConfig(BaseModel):
    """Configuration for the trained model to audit."""

    path: str = Field(..., description="Path to the .joblib model file.")
    backend: Literal["sklearn"] = Field(
        "sklearn", description="Model backend. Only 'sklearn' in v1."
    )


class FairnessConfig(BaseModel):
    """Fairness evaluation parameters."""

    sensitive_features: list[str] = Field(
        ...,
        description=(
            "Column names in the dataset to treat as sensitive attributes. "
            "Must be non-empty and contain no duplicates."
        ),
    )
    fairness_metrics: list[FairnessMetric] = Field(
        default_factory=lambda: [
            FairnessMetric.demographic_parity_difference,
            FairnessMetric.equalized_odds_difference,
            FairnessMetric.selection_rate_ratio,
        ],
        description="Fairness metrics to compute. Defaults to DPD, EOD, SRR.",
    )
    parity_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-metric fail-boundary overrides. Keys must be valid FairnessMetric "
            "names; values must be > 0. The warn boundary is always re-derived "
            "automatically (see docs/FAIRNESS_METRICS.md)."
        ),
    )
    base_metric: str = Field(
        "accuracy",
        description=(
            "Base performance metric used in the primary chart and subgroup breakdown. "
            f"Allowed values: {sorted(_ALLOWED_BASE_METRICS)}. Extended in v1.1."
        ),
    )
    intersectional: bool = Field(
        False,
        description=(
            "If True, compute metrics on cross-product groups of all sensitive "
            "features. Default False — individual feature groups only."
        ),
    )

    @field_validator("sensitive_features")
    @classmethod
    def sensitive_features_non_empty_no_duplicates(cls, v: list[str]) -> list[str]:
        """Validate that sensitive_features is non-empty and duplicate-free.

        Args:
            v: The raw sensitive_features list.

        Returns:
            Validated list.

        Raises:
            ValueError: If the list is empty or contains duplicates.
        """
        if not v:
            raise ValueError("sensitive_features must be non-empty")
        if len(v) != len(set(v)):
            raise ValueError("sensitive_features must not contain duplicates")
        return v

    @field_validator("parity_thresholds")
    @classmethod
    def validate_parity_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate parity_thresholds keys and values.

        Args:
            v: The raw parity_thresholds dict.

        Returns:
            Validated dict.

        Raises:
            ValueError: If any key is not a valid FairnessMetric or any value
                is not positive.
        """
        valid_keys = {m.value for m in FairnessMetric}
        for key, val in v.items():
            if key not in valid_keys:
                raise ValueError(
                    f"parity_thresholds key {key!r} is not a valid FairnessMetric. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            if val <= 0:
                raise ValueError(
                    f"parity_thresholds[{key!r}] must be > 0, got {val}"
                )
        return v

    @field_validator("fairness_metrics")
    @classmethod
    def fairness_metrics_non_empty(cls, v: list[FairnessMetric]) -> list[FairnessMetric]:
        """Ensure at least one metric is requested.

        Args:
            v: The raw fairness_metrics list.

        Returns:
            Validated list.

        Raises:
            ValueError: If the list is empty.
        """
        if not v:
            raise ValueError("fairness_metrics must contain at least one metric")
        return v

    @field_validator("base_metric")
    @classmethod
    def validate_base_metric(cls, v: str) -> str:
        """Validate that base_metric is one of the supported values.

        Args:
            v: The raw base_metric string.

        Returns:
            Validated base_metric.

        Raises:
            ValueError: If v is not in _ALLOWED_BASE_METRICS.
        """
        if v not in _ALLOWED_BASE_METRICS:
            raise ValueError(
                f"base_metric must be one of {sorted(_ALLOWED_BASE_METRICS)}, got {v!r}"
            )
        return v


class RunConfig(BaseModel):
    """Top-level configuration for a single fairness audit run."""

    run_name: str = Field(..., description="Human-readable name for this run.")
    description: str | None = Field(
        None, description="Optional description of this audit run."
    )
    dataset: DatasetConfig
    model: ModelConfig
    fairness: FairnessConfig

    @field_validator("run_name")
    @classmethod
    def run_name_non_empty(cls, v: str) -> str:
        """Strip and validate that run_name is non-empty.

        Args:
            v: Raw run_name value.

        Returns:
            Stripped run_name.

        Raises:
            ValueError: If run_name is blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("run_name must not be blank")
        return stripped
