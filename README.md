# ProForma-20Q

**A public benchmark for joint probabilistic forecasting of complete quarterly
financial statements at horizons 1–20 quarters.**

ProForma-20Q defines a *task*, not a model. Given your own WRDS credentials, one
command rebuilds the exact data environment and evaluation used in the companion
ICAIF '26 paper; you then train *your* model and score a forecast file with our
CLI. It is usable by someone who has never heard of the Forma model — build the
data, train, submit.

> **No data is distributed here.** The repository contains code, configs, and
> checksums only. WRDS / Compustat / CRSP credentials and license are entirely
> the user's responsibility.

---

## What it is

- **Universe.** Compustat Fundamentals Quarterly (`comp.fundq`) via WRDS,
  1970–2024, financial firms (SIC 6000–6999) excluded.
- **Targets.** The **78 `pf_full` statement items** — the full income statement,
  balance sheet, and cash-flow statement — forecast **jointly** at horizons
  **h = 1..20** quarters. No firm identifiers are used as features
  (universal-forecaster rule).
- **Splits (by calendar time).** train 1971–2001 / val 2002–2009 / test 2010–2024.
- **Transform.** Every item is scaled by a firm-size proxy, passed through
  `asinh`, z-scored per item, and clamped to `|z| ≤ 6`. All forecasting and
  scoring happen in this regularized space. (The transform is `asinh`, *not*
  signed-log — some legacy docs are wrong.)
- **Two data views.** A **tabular** lagged feature matrix (4 recent levels + 8
  YoY changes per item, plus FF48 industry dummies) and a **tuple**
  `(account, quarter, value)` sparse form. Both builders ship.
- **Evaluation.** Change-space R² (shared denominator), MAE/MSE, and — for
  forecasts that carry a `sigma` — NLL, CRPS, and calibration (z2, cover95), on a
  strict all-models common sample. See [SUBMISSION.md](SUBMISSION.md).

---

## Install

```bash
pip install proforma-20q            # scoring + baselines (any modern Python 3.10+)
pip install proforma-20q[wrds]      # also the WRDS client, for the data build
```

or from source:

```bash
git clone https://github.com/forma-lab-mccombs/proforma-20q
cd proforma-20q
pip install -e .[wrds,dev]
```

---

## Quickstart

### 1. Build the data (needs WRDS)

```bash
proforma20q build --wrds-user <your_wrds_user>
# -> prompts for your WRDS password (via ~/.pgpass or the wrds client)
# -> downloads Compustat+CRSP, processes both views, verifies checksums
```

This writes the tabular and tuple artifacts to `data/processed/`. The
**tabular test file is also the evaluation ground truth** — there is no separate
truth file.

If you already have the raw panel, skip the download:

```bash
proforma20q build --raw data/raw/compustat_with_permno.parquet
```

### 2. Reproduce the reference baselines (optional but recommended)

```bash
proforma20q baselines            # writes results/forecasts/{naive,fade,elasticnet,linear}__predictions.parquet
proforma20q evaluate --against baselines --out results/metrics
```

Confirm your build reproduces the published baseline numbers before trusting your
own model's score.

### 3. Score your model

Produce a forecast file in the [submission format](SUBMISSION.md), then:

```bash
proforma20q validate my_forecasts.parquet
proforma20q evaluate my_forecasts.parquet --against baselines --out results/metrics
```

### Library API

```python
from proforma20q.build import build
from proforma20q.baselines import run_baselines
from proforma20q.evaluate import evaluate_forecasts
from proforma20q.baselines.common import load_tabular

truth = load_tabular("data/processed/tabular_test__pf_full__r13_node_optionD_indfe_val8.parquet")
result = evaluate_forecasts({"mine": "my_forecasts.parquet"}, truth)
print(result.leaderboard("r2"))
```

---

## Reference baselines

Cheap to run, they anchor the leaderboard and let you verify your pipeline:

| baseline     | what it is |
|--------------|------------|
| `naive`      | random walk — forecast = value at origin; the **change-space zero anchor** (R² ≈ 0 by construction). Every model worth shipping beats it. |
| `fade`       | pooled AR(1) / fade-to-mean — one `(ρ_h, b_h)` per horizon, pooled across items and firms. |
| `elasticnet` | per-horizon `(alpha, l1_ratio)` cross-validated on `niq` over 2002–2009 and reused across all 78 targets; refits on train+val. |
| `linear`     | plain OLS per (target, horizon). |

**The Forma model is not included** — it is released separately.

## Leaderboard

| model | R² (global) | MAE | NLL | CRPS |
|-------|:-----------:|:---:|:---:|:----:|
| naive (RW anchor) | 0.000 | — | — | — |
| fade / AR(1) | *tbd* | — | — | — |
| ElasticNet | *tbd* | — | — | — |
| *your model here* | | | | |

Numbers are populated from the canonical build at release. To add a model, open a
PR with your forecast-file checksum and reproduction instructions.

---

## Reproducing the canonical build

Compustat is **revised over time**, so a fresh WRDS pull today will not be
bit-identical to the canonical snapshot. We therefore:

1. **Pin the environment.** For a bit-exact rebuild, install the lockfile:
   ```bash
   pip install -r requirements-lock.txt
   ```
2. **Pin the query.** The exact WRDS SQL lives in `src/proforma20q/download.py`;
   the sample window and filters are in `src/proforma20q/configs/task.yaml`.
3. **Publish hashes, not data.** `src/proforma20q/checksums.json` holds the md5s
   of the canonical artifacts (plus per-column hashes of the regularized tabular
   columns). `proforma20q build` verifies them automatically.
4. **Quantify drift instead of hard-failing.** If your vintage differs:
   ```bash
   proforma20q build --report-drift        # or: proforma20q report-drift
   ```
   reports, per artifact, the fraction of columns that diverge from the canonical
   checksums — so you can see *how much* your Compustat vintage moved without ever
   needing (or exposing) the canonical data.

**Definition of done for a reproduction:** a fresh machine + WRDS login →
`build` completes → checksums match (or the drift report explains the delta) →
`evaluate` on the shipped baseline forecasts reproduces the published
naive / fade / ElasticNet numbers.

---

## Reproducing the paper's pooled tables

`evaluate` scores on the **all-submitted-models common sample** (a cell counts
only if every model you pass predicts it finitely). The paper's headline table
instead scores each column on a **fixed pooled sample** — the common sample of a
specific set of models, so that adding a new entrant never moves a published
number. Its primary column is the **Full sample: 327,244,429 cells**.

`--sample-mask` reproduces that. Pass a mask of the pooled cells and every metric
is computed on `mask ∩ your models' common sample`:

```bash
proforma20q evaluate my_forecasts.parquet --against baselines \
    --sample-mask full_sample_mask.parquet
```

The mask is a keys table (`firm, target, origin, horizon`) or a compact
grid-aligned bit array (`.npy`, `np.packbits` of the canonical cell order — no
firm identifiers). Because the reference model's coverage is the binding
constraint, the Full mask is exactly *its finite-prediction cells ∩ truth*;
`scripts/build_full_sample_mask.py` rebuilds and verifies it (`--expect
327244429`). The published mask ships as a release asset (66–136 MB), with its
md5 in the repo.

Restricted to the Full mask, the shipped baselines reproduce the paper's Panel A
Full column **to the digit** (all on the identical 327,244,429-cell sample):

| baseline | R² (this repo) | R² (paper) |
|---|:--:|:--:|
| naive | −0.0412 | −0.041 |
| fade / AR(1) | 0.1827 | 0.183 |
| ElasticNet | 0.2585 | 0.258 |

**Sub-sample analysis.** The same flag scores any narrower pool — e.g. a mask
intersected with a model that only covers 25 targets (the paper's GBM column,
109 M cells) or a fixed-budget origin subsample (the LLM column, 2.15 M cells).
Those masks are *not* part of standard scoring; they are examples of the
sub-sample analyses the package enables for entrants with structural or budgetary
coverage limits.

---

## Task definition = single source of truth

Everything that defines the benchmark is in
[`src/proforma20q/configs/task.yaml`](src/proforma20q/configs/task.yaml)
(splits, targets, horizons, transform hyperparameters, size proxy, industry FE),
with the 78-item universe in `feature_sets.yaml` and the FF48 SIC ranges in
`ff48_sic_ranges.json`. Changing any of these defines a *different* benchmark and
invalidates published checksums and leaderboard numbers.

## Probabilistic track status

The point-forecast metrics (R²/MAE/MSE) and the single-family NLL/CRPS/calibration
metrics are final. Mixture NLL, PIT histograms, and Diebold–Mariano tests are
being finalized in the main research repo and will be ported here to match the
paper's evaluation to the digit — tracked, not forked.

## License

Code is licensed **Apache-2.0** (see [LICENSE](LICENSE) / [NOTICE](NOTICE)). **No
data is distributed**; you are responsible for your WRDS/Compustat/CRSP license.
The FF48 industry classification derives from the Kenneth R. French Data Library.

If you use ProForma-20Q, please cite it — see [CITATION.cff](CITATION.cff).
