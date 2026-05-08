"""Generate the synthetic EU energy burden dataset for smoke-testing.

Outputs examples/data/energy_burden_synthetic.csv (5000 rows, seed=42).

Schema mirrors EU-SILC energy-poverty indicators with EnergyGuard partner
countries (IT, PT, LV, ES). Deliberate disparity is injected so that a
baseline LogisticRegression trained on the data will fail the fairness audit:
  - equalized_odds_difference > 0.10
  - selection_rate_ratio (best vs worst income decile) < 0.80
  - accuracy gap between deciles 1-3 and 8-10 > 0.10

Usage:
    python examples/scripts/generate_synthetic_energy_burden.py
"""
# NOTE on disparity calibration (v1):
# The current per-decile spend multipliers are deliberately strong
# (approx 2.2x for decile 1 down to 0.70x for decile 10) to produce a
# clear, unambiguous disparity signal in the smoke test. This
# results in burden rates of ~95% for decile 1 and ~0% for deciles
# 8-10, which is more extreme than real-world energy-poverty
# surveys (e.g., EU-SILC typically shows decile 1 ~30-50%,
# decile 10 ~3-10%).
#
# The extreme disparity is acceptable for v1 because:
#   - The smoke test asserts specific FAIL statuses on fairness
#     metrics; subtle disparity would make assertions flaky.
#   - It exercises edge cases (groups with 0 positive examples,
#     near-zero selection rates) that the engine must handle.
#
# In v1.1, recalibrate to realistic ranges once the engine and
# report layer are stable. Target: decile 1 ~50%, decile 10 ~5%.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
N_ROWS = 5000
OUT_PATH = Path("examples/data/energy_burden_synthetic.csv")

# Partner countries with population weights
COUNTRIES = ["IT", "ES", "PT", "LV"]
COUNTRY_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

# Energy cost multipliers per country (LV and PT structurally higher)
COUNTRY_ENERGY_MULTIPLIER = {"IT": 1.00, "ES": 0.95, "PT": 1.12, "LV": 1.20}

URBAN_RURAL = ["urban", "suburban", "rural"]
URBAN_RURAL_WEIGHTS = [0.45, 0.35, 0.20]

AGE_BANDS = ["<35", "35-65", ">65"]
AGE_BAND_WEIGHTS = [0.25, 0.55, 0.20]
AGE_ENERGY_MULTIPLIER = {"<35": 0.95, "35-65": 1.00, ">65": 1.10}

DWELLING_TYPES = ["apartment", "house", "other"]
DWELLING_TYPE_WEIGHTS = [0.55, 0.35, 0.10]

DWELLING_AGE_BANDS = ["<20yr", "20-50yr", ">50yr"]
DWELLING_AGE_WEIGHTS = [0.20, 0.50, 0.30]

EPC_CLASSES = list("ABCDEFG")
EPC_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.15]
# Higher EPC class letter → higher energy consumption factor
EPC_EFFICIENCY_FACTOR = {c: 0.4 + 0.2 * i for i, c in enumerate("ABCDEFG")}

HEATING_FUELS = ["gas", "electric", "oil", "wood", "district"]
FUEL_WEIGHTS = [0.40, 0.25, 0.10, 0.15, 0.10]
FUEL_FACTOR = {"gas": 1.00, "electric": 1.10, "oil": 0.95, "wood": 0.80, "district": 0.90}

# Base annual income (EUR) by income decile — decile 1 is lowest
DECILE_BASE_INCOME = {
    1: 8_000, 2: 11_000, 3: 14_000, 4: 17_500, 5: 21_000,
    6: 25_000, 7: 30_000, 8: 36_000, 9: 43_000, 10: 55_000,
}

BURDEN_THRESHOLD = 0.10  # energy_spend / income > 10% → is_energy_burdened


def _generate(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the dataset using a fixed RNG for reproducibility.

    Args:
        rng: Seeded numpy Generator.

    Returns:
        DataFrame with 5000 rows conforming to the energy burden schema.
    """
    n = N_ROWS

    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    urban_rural = rng.choice(URBAN_RURAL, size=n, p=URBAN_RURAL_WEIGHTS)
    income_decile = rng.integers(1, 11, size=n)  # uniform over [1, 10]
    age_head_band = rng.choice(AGE_BANDS, size=n, p=AGE_BAND_WEIGHTS)
    dwelling_type = rng.choice(DWELLING_TYPES, size=n, p=DWELLING_TYPE_WEIGHTS)
    dwelling_age_band = rng.choice(DWELLING_AGE_BANDS, size=n, p=DWELLING_AGE_WEIGHTS)
    building_efficiency_class = rng.choice(EPC_CLASSES, size=n, p=EPC_WEIGHTS)
    heating_fuel = rng.choice(HEATING_FUELS, size=n, p=FUEL_WEIGHTS)

    household_size = np.clip(rng.poisson(2.5, n), 1, 7)

    base_income = np.array([DECILE_BASE_INCOME[d] for d in income_decile], dtype=float)
    annual_income_eur = base_income * rng.lognormal(0.0, 0.15, n)

    country_mult = np.array([COUNTRY_ENERGY_MULTIPLIER[c] for c in country])
    age_mult = np.array([AGE_ENERGY_MULTIPLIER[a] for a in age_head_band])
    epc_factor = np.array([EPC_EFFICIENCY_FACTOR[c] for c in building_efficiency_class])
    fuel_factor = np.array([FUEL_FACTOR[f] for f in heating_fuel])
    size_factor = 0.7 + 0.1 * household_size

    # Fuel-poverty-trap disparity: low-income households pay proportionally more
    # (older housing stock, inability to invest in efficiency, worse fuel tariffs).
    # Index 0 unused; indices 1-10 map directly to income_decile values.
    _decile_penalty = np.array(
        [0.0, 2.2, 1.9, 1.6, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    )
    income_penalty = _decile_penalty[income_decile]

    annual_energy_spend_eur = (
        epc_factor * fuel_factor * size_factor * 900.0
        * country_mult * age_mult * income_penalty
        * rng.lognormal(0.0, 0.10, n)
    )

    is_energy_burdened = (
        annual_energy_spend_eur / annual_income_eur > BURDEN_THRESHOLD
    ).astype(int)

    return pd.DataFrame({
        "household_id": np.arange(1, n + 1),
        "country": country,
        "urban_rural": urban_rural,
        "income_decile": income_decile,
        "age_head_band": age_head_band,
        "household_size": household_size,
        "dwelling_type": dwelling_type,
        "dwelling_age_band": dwelling_age_band,
        "building_efficiency_class": building_efficiency_class,
        "heating_fuel": heating_fuel,
        "annual_income_eur": annual_income_eur.round(2),
        "annual_energy_spend_eur": annual_energy_spend_eur.round(2),
        "is_energy_burdened": is_energy_burdened,
    })


def main() -> None:
    """Entry point: generate dataset and write to OUT_PATH."""
    rng = np.random.default_rng(SEED)
    df = _generate(rng)

    print("\n=== Burden rate by income_decile ===")
    print(
        df.groupby("income_decile")["is_energy_burdened"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "burden_rate"})
        .to_string(float_format="{:.3f}".format)
    )

    print("\n=== Burden rate by country ===")
    print(
        df.groupby("country")["is_energy_burdened"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "burden_rate"})
        .to_string(float_format="{:.3f}".format)
    )

    print("\n=== Burden rate by age_head_band ===")
    print(
        df.groupby("age_head_band")["is_energy_burdened"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "burden_rate"})
        .to_string(float_format="{:.3f}".format)
    )

    rates = df.groupby("income_decile")["is_energy_burdened"].mean()
    for d in [1, 2, 3]:
        r = float(rates[d])
        if r <= 0.40:
            raise AssertionError(
                f"Decile {d} burden rate {r:.3f} ≤ 0.40 — disparity injection failed."
            )
    for d in [8, 9, 10]:
        r = float(rates[d])
        if r >= 0.15:
            raise AssertionError(
                f"Decile {d} burden rate {r:.3f} ≥ 0.15 — high-income households over-burdened."
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(df), OUT_PATH)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
