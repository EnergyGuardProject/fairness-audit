"""TaskStrategy ABC and concrete classification strategy.

Pattern mirrors the robustness service. pipeline.py selects the strategy
based on the task type inferred from RunConfig (v1: always classification).

Extension points:
    v1.1 — RegressionFairnessStrategy
    v1.2 — SubgroupPerformanceStrategy
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from runner.config.models import RunConfig

logger = logging.getLogger(__name__)


class TaskStrategy(ABC):
    """Abstract strategy for executing a fairness evaluation task."""

    @abstractmethod
    def execute(self, config: RunConfig, job_dir: Path) -> None:
        """Execute the evaluation task and write all artifacts to job_dir.

        Args:
            config: Validated RunConfig for this evaluation.
            job_dir: Directory in which to write metrics.json,
                metrics_user.json, and report.html.

        Raises:
            NotImplementedError: Concrete strategies must implement this.
        """
        ...


class ClassificationFairnessStrategy(TaskStrategy):
    """Strategy for binary and multiclass classification fairness evaluation.

    Sequence:
        1. Load model via ModelAdapter
        2. Load dataset via CsvDatasetLoader
        3. Predict on full dataset
        4. Infer n_classes from y_true.nunique()
        5. FairnessEngine.run(y_true, y_pred, sensitive, n_classes)
        6. Apply thresholds → evaluate statuses
        7. Build two-tier output (metrics.json, metrics_user.json)
        8. Render HTML report
        9. Write all artifacts to job_dir
    """

    def execute(self, config: RunConfig, job_dir: Path) -> None:
        """Run classification fairness evaluation end-to-end.

        Args:
            config: Validated RunConfig.
            job_dir: Target directory for all output artifacts.

        Raises:
            ValueError: If requested metrics are incompatible with n_classes.
            FileNotFoundError: If model or dataset path does not exist.
        """
        raise NotImplementedError
