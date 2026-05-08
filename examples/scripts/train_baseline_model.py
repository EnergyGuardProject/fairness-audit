"""Train a baseline LogisticRegression model on the synthetic energy burden dataset.

Produces examples/models/baseline_logreg.joblib — a sklearn Pipeline containing
a ColumnTransformer (OrdinalEncoder + StandardScaler) and LogisticRegression.
The model is trained on ALL 5000 rows: the fairness audit measures disparity in
model behaviour across groups, not test-set generalisation.

Prerequisites:
    Run generate_synthetic_energy_burden.py first.

Usage:
    python examples/scripts/train_baseline_model.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

DATA_PATH = Path("examples/data/energy_burden_synthetic.csv")
MODEL_PATH = Path("examples/models/baseline_logreg.joblib")
TARGET_COLUMN = "is_energy_burdened"
DROP_COLUMNS = ["household_id"]

CATEGORICAL_FEATURES = [
    "country", "urban_rural", "income_decile",
    "age_head_band", "dwelling_type", "dwelling_age_band",
    "building_efficiency_class", "heating_fuel",
]
NUMERIC_FEATURES = ["household_size", "annual_income_eur", "annual_energy_spend_eur"]

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """Construct the sklearn Pipeline (encoder + classifier).

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE
        )),
    ])


def main() -> None:
    """Load dataset, train pipeline, save to MODEL_PATH.

    Raises:
        FileNotFoundError: If DATA_PATH does not exist (run generator first).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run generate_synthetic_energy_burden.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLUMNS)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    overall_acc = (preds == y).mean()
    print(f"\nOverall accuracy (train): {overall_acc:.4f}")

    print(f"\n{'decile':>8}  {'accuracy':>10}  {'selection_rate':>14}  {'n':>6}")
    for decile in sorted(df["income_decile"].unique()):
        mask = df["income_decile"] == decile
        acc = (preds[mask] == y[mask]).mean()
        sel = preds[mask].mean()
        print(f"{decile:>8}  {acc:>10.4f}  {sel:>14.4f}  {mask.sum():>6}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Saved pipeline to %s", MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
