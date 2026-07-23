# ProForma-20Q submission format

A submission is a **single parquet file**: one row per forecasted cell. You
produce it by training your model on the ProForma-20Q build and predicting the
78 `pf_full` statement items at horizons 1–20 for every test-set firm-quarter.

## Schema

| column       | type            | required | meaning |
|--------------|-----------------|:--------:|---------|
| `firm`       | string          | ✓ | firm identifier — the gvkey string from the truth file (e.g. `"001045"`) |
| `target`     | string          | ✓ | statement item, one of the 78 `pf_full` targets (e.g. `"niq"`) |
| `origin`     | timestamp / Period[Q] | ✓ | the base quarter *t* (last quarter observed before forecasting) |
| `horizon`    | int (1..20)     | ✓ | forecast is for quarter *t + horizon* |
| `prediction` | float           | ✓ | point forecast of the item at *t+h*, **in regularized space** |
| `sigma`      | float (> 0)     | optional | predictive standard deviation, same regularized space (for `student_t`, the t **scale** parameter — see *Density family*) — include it to enter the **probabilistic track** |

Notes:

- **Everything is in the regularized target space** defined by the build
  (`scale → asinh → per-item z-score → clamp |z| ≤ 6`). Predictions are the
  **level** at *t+h* (not the change); the evaluator forms the change internally
  as `prediction − {target}_level_0`.
- Write with **`engine="fastparquet"`** (never pyarrow). The helper
  `proforma20q.schema.write_forecast` does this and downcasts the float payload
  to float32.
- The internal Forma column names `firm_id` / `quarter` / `forecast_horizon` are
  accepted as aliases and normalized on read, so research-repo forecast files
  score without conversion.
- One row per `(firm, target, origin, horizon)`. Duplicates keep the first;
  missing cells are simply absent (they drop out of the common sample).

## Density family (probabilistic track)

`sigma` is interpreted under a predictive **family**, defaulting to Gaussian. To
declare Laplace or Student-t, drop a sidecar `myforecast.nll.json` next to the
parquet:

```json
{ "family": "student_t", "df": 5.0 }
```

`family ∈ {gaussian, laplace, student_t}`. For **gaussian** and **laplace**,
`sigma` is the predictive **standard deviation** (Laplace's scale `b = sigma/√2`
is derived internally, so its SD is `sigma`). For **student_t**, `sigma` is the t
**scale** parameter directly — the standard deviation `sigma·√(ν/(ν−2))` is finite
only for `ν > 2`, so the scale (not the SD) is the primary parameter; the
evaluator uses this same scale for both NLL and CRPS, so the two scores describe
the identical predictive density. NLL and CRPS are always computed **by the
evaluator** against the single shared ground truth — never trusted from the
generator.

## Minimal example

```python
import pandas as pd
from proforma20q.schema import write_forecast

fc = pd.DataFrame({
    "firm":       ["001045", "001045", "001045"],
    "target":     ["niq",    "niq",    "revtq"],
    "origin":     pd.Timestamp("2011-12-31"),
    "horizon":    [1, 2, 1],
    "prediction": [0.42, 0.55, -1.13],
    "sigma":      [0.8, 0.9, 0.7],   # optional
})
write_forecast(fc, "my_forecasts.parquet")   # validates + writes with fastparquet
```

Score it:

```bash
proforma20q validate my_forecasts.parquet
proforma20q evaluate my_forecasts.parquet --against baselines
```

A ready-made **synthetic** example lives in [`examples/`](examples/)
(`example_forecast.parquet` + a matching `example_truth.parquet`); regenerate it
with `python examples/make_example.py`.

## What the evaluator reports

Per aggregation level (`global` / `by_target` / `by_horizon` /
`by_target_horizon`):

- **R²** in change space, against the variance of realized changes, with a
  denominator **shared across all compared models**.
- **MAE**, **MSE** (identical in levels and changes).
- **NLL**, **CRPS**, **z2** (calibration), **cover95** — only for the rows your
  forecast carried a valid `sigma`.
- `avg_obs`, `n_complete` — coverage diagnostics.

A cell scores for everyone only if the truth, the baseline, and **every** compared
model's prediction are all finite (the strict common-sample inner join); a
self-check asserts each model's `n_metric` equals the block's `n_complete`.
