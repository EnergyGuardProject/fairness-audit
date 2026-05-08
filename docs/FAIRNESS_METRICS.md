# EnergyGuard Fairness Audit — Metrics Reference

All metrics are computed via Fairlearn's `MetricFrame`. This document defines every
metric, its threshold boundaries, and the headline scoring rubric.

---

## Primary Metrics (thresholded)

These appear in `summary.primary_metrics` with a `status` of pass / warn / fail.

### Demographic Parity Difference (DPD)

```
DPD = max(selection_rate_by_group) − min(selection_rate_by_group)
```

Measures the largest gap in positive prediction rates between any two groups.
`0` = perfectly equal treatment. `direction: lower_is_better`.

| Status | Boundary |
|--------|----------|
| pass   | DPD ≤ 0.05 |
| warn   | 0.05 < DPD ≤ 0.10 |
| fail   | DPD > 0.10 |

### Equalized Odds Difference (EOD)

```
EOD = max(
    max(FPR_by_group) − min(FPR_by_group),
    max(FNR_by_group) − min(FNR_by_group),
)
```

Measures the largest discrepancy in error rates (false positive or false negative)
across groups. Binary classification only. `direction: lower_is_better`.

| Status | Boundary |
|--------|----------|
| pass   | EOD ≤ 0.05 |
| warn   | 0.05 < EOD ≤ 0.10 |
| fail   | EOD > 0.10 |

### Selection Rate Ratio (SRR)

```
SRR = min(selection_rate_by_group) / max(selection_rate_by_group)
```

Ratio of the lowest to the highest positive prediction rate across groups.
`1.0` = perfectly equal; lower values indicate greater disparity.
`direction: higher_is_better`. Not a native Fairlearn function; computed from
MetricFrame's per-group selection rates. Emits `0.0` and logs a warning if the
highest group rate is zero.

| Status | Boundary |
|--------|----------|
| pass   | SRR ≥ 0.90 |
| warn   | 0.80 ≤ SRR < 0.90 |
| fail   | SRR < 0.80 |

---

## Diagnostic Metrics (not thresholded)

These appear in `subgroup_breakdown` and `metrics.json` but carry no pass/warn/fail
status. They are informational only and do not affect the headline score.

| Metric | Formula |
|--------|---------|
| `accuracy` | correct predictions / total predictions per group |
| `selection_rate` | positive predictions / total predictions per group |
| `false_positive_rate` | FP / (FP + TN) per group (binary only) |
| `false_negative_rate` | FN / (FN + TP) per group (binary only) |

---

## Threshold Override

Users may override the **fail boundary** for any primary metric via
`fairness.parity_thresholds` in the YAML config:

```yaml
fairness:
  parity_thresholds:
    demographic_parity_difference: 0.15   # relaxed threshold for this dataset
```

The **warn boundary** is always re-derived automatically:

- **Difference metrics** (lower_is_better): `warn = fail / 2`
- **Ratio metrics** (higher_is_better): `warn = (1.0 + fail) / 2`

Rationale for ratio warn derivation: applying `warn = fail / 2` to ratio metrics
would give `warn = 0.40` for `fail = 0.80`, which represents a severely unfair
model. Using the midpoint between 1.0 (perfect equality) and the fail boundary
produces a warn zone proportional to the acceptable range.

---

## Headline Scoring Rubric

Each primary metric contributes equally to the headline score:

```
headline_score = mean(
    1.0 if status == "pass" else
    0.5 if status == "warn" else
    0.0
    for metric in summary.primary_metrics
)
```

Labels are derived from status composition, not from the numeric score:

| Label      | Condition |
|------------|-----------|
| `good`     | All primary metrics pass |
| `moderate` | At least one warn; no fails |
| `poor`     | At least one fail |

---

## Metric Compatibility

| Metric | Binary | Multiclass |
|--------|--------|-----------|
| `demographic_parity_difference` | ✓ | ✓ |
| `selection_rate_ratio` | ✓ | ✓ |
| `equalized_odds_difference` | ✓ | ✗ (raises `ValueError`) |
| `false_positive_rate` (diagnostic) | ✓ | ✗ |
| `false_negative_rate` (diagnostic) | ✓ | ✗ |

The engine checks compatibility at entry (`n_classes` inferred from
`y_true.nunique()`). Incompatible combinations fail loudly before any computation
begins.

---

## Intersectional Mode

When `fairness.intersectional: true`, MetricFrame receives sensitive features as a
DataFrame, producing groups for all cross-product combinations (e.g., `income_decile=1,
country=LV`). This can produce many groups; only the top two features by accuracy
range are shown in the tertiary heatmap chart when ≥ 3 features are present.

A `HEATMAP_FEATURE_SELECTION` info-level warning is emitted in the report naming the
selected and omitted features.

Intersectional mode is **off by default** (`intersectional: false`) because small
intersectional groups produce unstable metric estimates.
