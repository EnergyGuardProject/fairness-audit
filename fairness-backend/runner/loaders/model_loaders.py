"""Model loaders for the fairness audit runner.

v1 supports sklearn only (.joblib files). Future backends (PyTorch, MLflow)
are deferred to v2 / v1.1 respectively.
"""
from __future__ import annotations

import logging
from pathlib import Path

from runner.loaders.model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


def load_sklearn_model(path: Path) -> ModelAdapter:
    """Load a joblib-serialised sklearn model and wrap it in a ModelAdapter.

    Args:
        path: Path to the .joblib file.

    Returns:
        ModelAdapter wrapping the loaded model.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file cannot be deserialised as a sklearn model.
    """
    import joblib  # type: ignore[import-untyped]

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Cannot deserialise model from {path}: {exc}") from exc
    logger.info("Loaded sklearn model from %s (type: %s)", path, type(model).__name__)
    return ModelAdapter(model)
