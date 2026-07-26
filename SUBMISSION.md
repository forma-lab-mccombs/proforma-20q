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
  to float32; for a full-coverage submission use
  `proforma20q.schema.write_forecast_blocks` instead (see *How big a submission
  is*).
- The internal Forma column names `firm_id` / `quarter` / `forecast_horizon` are
  accepted as aliases and normalized on read, so research-repo forecast files
  score without conversion.
- One row per `(firm, target, origin, horizon)`. Duplicates keep the first;
  missing cells are simply absent (they drop out of the common sample).

## How big a submission is

Plan for the size before you write the file. On the canonical build the test
split has **352,106 firm-quarters**, so **full coverage is**

```
352,106 origins × 78 targets × 20 horizons = 549,285,360 rows
```

Measured: **3.54 GB on disk** for the point track (float32 predictions; add
~2 GB if you carry `sigma`), and **~73 GB as a single in-memory frame** — the
`firm` and `target` string columns cost ~112 of the 132 bytes per row. Building
it as one frame raises `ArrayMemoryError` well before that on an ordinary
machine. Partial coverage is allowed (missing
cells simply drop out of the common sample), so a subset of targets or horizons
is a legitimate, much smaller entry — but do not plan a full-coverage run around
a single in-memory frame.

**Write it in blocks.** `proforma20q.schema.write_forecast_blocks` takes any
iterable of submission-schema frames — typically one per `(target, horizon)` —
validates each and appends it as a parquet row-group, so the forecast itself is
never held in memory; what you pay for is your own model state plus a ~4M-row
write buffer (a few hundred MB). Measured end to end, the shipped `naive` and
`fade` baselines write their full 549,285,360-row forecasts at **11.2 GB peak**,
nearly all of which is the tabular splits they read, not the forecast they
write:

```python
from proforma20q.schema import write_forecast_blocks

def blocks(test_origins, model):
    for target in TARGETS:                       # 78
        for h in range(1, 21):                   # 20
            yield pd.DataFrame({
                "firm":       test_origins["firm"].to_numpy(),
                "target":     target,
                "origin":     test_origins["origin"].to_numpy(),
                "horizon":    h,
                "prediction": model.predict(test_origins, target, h),
            })

n = write_forecast_blocks(blocks(test_origins, model), "my_forecasts.parquet")
```

The shipped baselines use exactly this path
(`proforma20q.baselines.iter_baseline_blocks` → `write_forecast_blocks`).

Two properties worth knowing:

- **Every block is validated**, duplicate keys included. Rows accumulate in
  `<name>.parquet.partial` and are renamed into place only when the last block
  is written, so a run that dies at block 1,400 of 1,560 cannot leave a
  well-formed parquet that would score as an intentional partial-coverage entry.
- **`proforma20q validate` streams the finished file row-group by row-group**
  (it cannot load it — ~550M rows is ~73 GB as a frame, dominated by the `firm`
  and `target` string columns). It therefore checks every row-group in full but
  does **not** detect duplicate keys that span two row-groups. A writer that
  emits one row-group per `(target, horizon)` cannot produce them.

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
