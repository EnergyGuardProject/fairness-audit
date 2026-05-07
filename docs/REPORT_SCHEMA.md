# EnergyGuard Trustworthiness Report Schema v1.0

All T5.2 trustworthiness services (fairness, adversarial robustness, 
explainability) MUST produce a `metrics_user.json` artifact conforming 
to this schema. The EnergyGuard frontend renders any conforming payload 
with a single component tree.

One payload per submission per service.

## Top-level Structure

```json
{
  "schema_version": "1.0",
  "report_meta":        { ... },
  "evaluation_setup":   { ... },
  "summary":            { ... },
  "subgroup_breakdown": [ ... ],
  "charts":             { ... },
  "warnings":           [ ... ],
  "recommendations":    [ ... ]
}
```

---

## report_meta (required, identical across services)

```json
{
  "service": "fairness",
  "service_version": "0.1.0",
  "run_id": "20260507_143012_a1b2c3d4",
  "run_name": "string",
  "description": "string | null",
  "task_type": "classification" | "regression",
  "model_backend": "sklearn" | "pytorch" | "mlflow",
  "mlflow_run_id": "string | null",
  "mlflow_tracking_uri": "string | null",
  "dataset_uri": "string",
  "feature_count": 14,
  "sample_count": 32561,
  "samples_evaluated": 32561,
  "timestamp_utc": "2026-05-07T14:30:12Z",
  "status": "ok" | "partial" | "error"
}
```

---

## evaluation_setup (service-specific keys, common shape)

Service-specific keys go inside this block. Frontend renders as 
key-value table without interpretation.

Fairness shape:
```json
{
  "sensitive_features": ["sex", "race"],
  "fairness_metrics": [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "selection_rate_ratio"
  ],
  "parity_thresholds": {
    "demographic_parity_difference": 0.1,
    "equalized_odds_difference": 0.1,
    "selection_rate_ratio": 0.8
  },
  "base_metric": "accuracy"
}
```

---

## summary (required)

```json
{
  "headline_score": 0.58,
  "headline_label": "good" | "moderate" | "poor",
  "primary_metrics": [
    {
      "key": "demographic_parity_difference",
      "label": "Demographic Parity Difference",
      "value": 0.19,
      "format": "scalar" | "percent" | "ratio",
      "direction": "lower_is_better" | "higher_is_better",
      "threshold": 0.1,
      "status": "pass" | "warn" | "fail"
    }
  ]
}
```

`headline_score` is the single number the dashboard surfaces for this 
submission (0..1, higher = better). Each service defines its own scoring 
rubric and documents it in its FAIRNESS_METRICS.md.

---

## subgroup_breakdown (required, may be empty)

Generic row-wise breakdown. For fairness, rows are groups (combinations 
of sensitive feature values).

```json
[
  {
    "row_label": "sex=Male",
    "row_kind": "group",
    "metrics": {
      "accuracy": 0.83,
      "selection_rate": 0.31,
      "false_positive_rate": 0.10,
      "false_negative_rate": 0.41
    },
    "n_samples": 21790,
    "status": "pass" | "warn" | "fail"
  }
]
```

---

## charts (required, may have empty point arrays)

Three named slots. Forces curation. Frontend has one renderer per `type`.

```json
{
  "primary_chart": {
    "type": "bar" | "line" | "scatter" | "heatmap",
    "title": "Accuracy by Group",
    "x_label": "group",
    "y_label": "accuracy",
    "points": [["sex=Male", 0.83], ["sex=Female", 0.92]],
    "reference_value": 0.85,
    "annotations": []
  },
  "secondary_chart": { ... } | null,
  "tertiary_chart": { ... } | null
}
```

For fairness specifically:
- **primary_chart**: bar chart of base metric (accuracy/MAE) per group, 
  reference line at overall metric
- **secondary_chart**: bar chart of selection rate per group
- **tertiary_chart**: heatmap of disparity across sensitive feature 
  combinations (when ≥2 sensitive features), else null

---

## warnings (required, may be empty)

```json
[
  {
    "code": "BINARY_SENSITIVE_ONLY",
    "severity": "info" | "warning" | "error",
    "message": "Only one sensitive feature provided; intersectional disparities not assessed."
  }
]
```

---

## recommendations (required, may be empty)

```json
[
  {
    "priority": "high" | "medium" | "low",
    "category": "data" | "model" | "deployment" | "documentation",
    "action": "Investigate label distribution by sex; check for historical bias in income labelling.",
    "rationale": "Selection rate disparity of 0.20 likely reflects label bias more than model bias.",
    "external_refs": ["https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html"]
  }
]
```

---

## Fairness-Specific Headline Scoring (v1 rubric)

Each fairness metric in `summary.primary_metrics` carries a `status` 
(pass / warn / fail). Headline label is derived as:

- **good**: all metrics pass
- **moderate**: at least one warn, no fails
- **poor**: at least one fail

Headline score is computed as:
