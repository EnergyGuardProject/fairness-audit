"""Dataset loaders for the fairness audit runner.

v1 supports CSV only. Regression datasets are deferred to v1.1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from runner.config.models import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    """Parsed dataset split into features, labels, and sensitive columns.

    Attributes:
        X: Feature DataFrame (excludes target and sensitive feature columns
           that are not also features — in v1 sensitive features are kept in X
           since the model was trained with them).
        y_true: Binary target Series.
        sensitive: DataFrame containing only the sensitive feature columns.
        df_full: The original full DataFrame (for provenance reporting).
    """

    X: pd.DataFrame
    y_true: pd.Series  # type: ignore[type-arg]
    sensitive: pd.DataFrame
    df_full: pd.DataFrame


def load_csv_dataset(config: DatasetConfig, sensitive_features: list[str]) -> LoadedDataset:
    """Load and parse a CSV dataset according to DatasetConfig.

    Args:
        config: DatasetConfig with path and target_column.
        sensitive_features: Column names to extract as the sensitive DataFrame.
            All must exist in the CSV.

    Returns:
        LoadedDataset with X, y_true, sensitive, and df_full populated.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If target_column or any sensitive feature column is absent.
    """
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    if config.target_column not in df.columns:
        raise KeyError(
            f"Target column '{config.target_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    missing_sf = [col for col in sensitive_features if col not in df.columns]
    if missing_sf:
        raise KeyError(
            f"Sensitive feature columns not found in CSV: {missing_sf}. "
            f"Available columns: {list(df.columns)}"
        )

    y_true = df[config.target_column]
    X = df.drop(columns=[config.target_column])
    sensitive = df[sensitive_features].copy()

    logger.info(
        "Loaded dataset from %s: %d rows, %d feature columns",
        path, len(df), len(X.columns),
    )
    return LoadedDataset(X=X, y_true=y_true, sensitive=sensitive, df_full=df)
