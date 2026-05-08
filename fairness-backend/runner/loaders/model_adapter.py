"""ModelAdapter — uniform predict() interface over loaded model objects.

Wraps any model object (v1: sklearn only) behind a consistent interface so the
pipeline does not depend on the model type directly. Validation is duck-typed:
the adapter checks for a `predict` method and logs an info note if the object
is not a sklearn Pipeline (but does not fail).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Wraps a loaded model object with a uniform predict() interface."""

    def __init__(self, model: Any) -> None:
        """Validate and wrap a model object.

        Args:
            model: Any object with a `predict` method.

        Raises:
            TypeError: If model does not have a `predict` method.
        """
        if not hasattr(model, "predict"):
            raise TypeError(
                f"Loaded object of type {type(model).__name__!r} has no "
                "'predict' method and cannot be used as a model."
            )

        try:
            from sklearn.pipeline import Pipeline  # type: ignore[import]

            if not isinstance(model, Pipeline):
                logger.info(
                    "Loaded model is %s, not a sklearn Pipeline. "
                    "Ensure it accepts a pandas DataFrame as input.",
                    type(model).__name__,
                )
        except ImportError:
            pass

        self._model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[type-arg]
        """Generate hard-label predictions for input features.

        Args:
            X: Feature DataFrame. Column order must match what the model was
               trained on.

        Returns:
            1-D numpy array of predicted class labels.
        """
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:  # type: ignore[type-arg]
        """Generate probability predictions if the model supports them.

        Args:
            X: Feature DataFrame.

        Returns:
            2-D numpy array of shape (n_samples, n_classes), or None if the
            model does not have a `predict_proba` method.
        """
        raise NotImplementedError
