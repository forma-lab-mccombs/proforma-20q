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

Install from source — the supported path today (any modern Python 3.10+):

```bash
# from a download or clone of this repository:
cd proforma-20q
pip install -e .[wrds,dev]          # scoring + baselines + WRDS client + tests
```

Use `pip install -e .[wrds]` if you only need the data build, or
`pip install -e .` for the scoring/baseline path alone (no WRDS client).

> **PyPI — available once published.** On release the package will also install
> straight from PyPI. This is **not yet available** (publication happens when the
> repository goes public at submission):
>
> ```bash
> pip install proforma-20q          # scoring + baselines
> pip install proforma-20q[wrds]    # also the WRDS client, for the data build
> ```

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

**What it costs.** Measured on the canonical 1970–2024 panel (1,531,703 × 655),
`--which tabular --reg-stats canonical`, on a 34 GB Windows workstation:

| stage | peak RSS | wall |
|---|---|---|
| raw load + prep (YTD → computed items → scale) | 4.5 GB | ~20 s |
| wide-matrix build (1,173,598 × 2,547) | 13.7 GB | ~3 min |
| writing the three splits (~6.7 GB of parquet) | 20.3 GB | ~12 min |

The download pulls **82 of `comp.fundq`'s 648 columns** — the ones the benchmark
consumes — and does it in one-year chunks by default, cached under
`data/raw_chunks/` so an interrupted pull resumes. `--chunk-years 0` issues a
single query instead; `--all-columns` restores the old `SELECT f.*` (~7.9× more
data, ~100 GB peak over 1970–2024 — it does not complete on an ordinary
machine). `build --raw` applies the same projection when reading the panel, so
an existing `SELECT f.*` parquet costs no more than a projected one.

### 2. Reproduce the reference baselines

Do this before trusting your own model's score — it is how you find out whether
your build reproduces the published numbers. **Run the two cheap ones first**;
the linear family takes the better part of a day (see
[Reference baselines](#reference-baselines)).

```bash
proforma20q baselines --which naive,fade      # ~17 min, ~11 GB peak
proforma20q evaluate --against baselines --out results/metrics
```

```bash
proforma20q baselines                          # adds elasticnet + linear: hours
```

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

They anchor the leaderboard and let you verify your pipeline.

| baseline     | what it is | cost at canonical scale |
|--------------|------------|-------------------------|
| `naive`      | random walk — forecast = value at origin; the **change-space zero anchor** (R² ≈ 0 by construction). Every model worth shipping beats it. | measured: naive+fade together **17 min / 11 GB peak**, writing two 549,285,360-row forecasts (3.5 GB each) |
| `fade`       | pooled AR(1) / fade-to-mean — one `(ρ_h, b_h)` per horizon, pooled across items and firms. | ″ |
| `elasticnet` | per-horizon `(alpha, l1_ratio)` cross-validated on `niq` over 2002–2009 and reused across all 78 targets; refits on train+val. | **hours.** Measured one refit at canonical train size (578,831 × 985): 5.3 s at h=1, 1.8 s at h=20 → ~1–2 h for the 1,560 refits, plus ~1 h for the 840-fit CV grid (42 combos × 20 horizons) |
| `linear`     | plain OLS per (target, horizon). | **the most expensive of the four, not the cheapest.** Measured one fit at canonical train size: 32.5 s at h=1, 15.2 s at h=20 → **7–14 h** for 1,560 fits. OLS solves by SVD; coordinate descent with an L1 penalty is ~6× faster here |

Both linear-family baselines peaked at ~10 GB RSS in that measurement. For a
quick end-to-end pipeline check, run just the fast two:

```bash
proforma20q baselines --which naive,fade
```

**The Forma model is not included** — it is released separately.

## Leaderboard

The headline benchmark number is the **Full-sample pooled R²** — the paper's
327,244,429-cell column (see [Reproducing the paper's pooled tables](#reproducing-the-papers-pooled-tables)).
On a canonical build the shipped baselines score:

| model | R² (Full sample) |
|-------|:----------------:|
| naive (RW anchor) | −0.0412 |
| fade / AR(1) | 0.1827 |
| ElasticNet | 0.2585 |
| *your model here* | |

These are the same canonical values reproduced in the pooled table below;
regenerate them with a canonical build + `proforma20q evaluate --against baselines
--sample-mask …`. To add a model, open a PR with your forecast-file checksum and
reproduction instructions.

---

## Reproducing the canonical build

Compustat is **revised over time**, so a fresh WRDS pull today will not be
bit-identical to the canonical snapshot. We therefore:

1. **Pin the environment.** For a rebuild against pinned dependencies, install
   the lockfile:
   ```bash
   pip install -r requirements-lock.txt
   ```
   It pins the **direct** dependencies (no hashes; transitive deps float), so it
   reproduces the build to float precision rather than guaranteeing bit-for-bit
   identical bytes — use `report-drift` (below) to quantify any residual
   environment/vintage drift.
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
    --sample-mask artifacts/full_sample_mask_bits.npy
```

The mask is a keys table (`firm, target, origin, horizon`) or a compact
grid-aligned bit array (`.npy`, `np.packbits` of the canonical cell order — no
firm identifiers). **The prebuilt grid-aligned mask ships in this repository** at
[`artifacts/full_sample_mask_bits.npy`](artifacts/full_sample_mask_bits.npy)
(~66 MB; md5 `a36008d8…`, pinned in
[`scripts/full_sample_mask.manifest.json`](scripts/full_sample_mask.manifest.json)),
so the command above needs no download. Because the reference model's coverage is
the binding constraint, the Full mask is exactly *its finite-prediction cells ∩
truth*; `scripts/build_full_sample_mask.py` also rebuilds and verifies it
(`--expect 327244429`) offline from the Forma forecast, reproducing the same md5.

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

## Data & artifacts

**No WRDS-derived data is distributed** — you rebuild the tabular/tuple artifacts
yourself from your own Compustat licence (`proforma20q build`).

The **Full-sample mask ships in this repository** — it holds only a coverage
bitmap, no firm-level values:

| artifact | size | md5 | for |
|---|---|---|---|
| [`artifacts/full_sample_mask_bits.npy`](artifacts/full_sample_mask_bits.npy) | 66 MB | `a36008d8…` | the 327,244,429-cell Full-sample mask (grid-aligned packbits, no firm ids); pass to `evaluate --sample-mask`. |

One further artifact is **not yet available** and is deferred to publication:

> **⚠ Not yet available (placeholder).** The Forma ensemble forecast below is
> published to an archival record (DOI assigned **on publication**); no record
> exists yet. Until then `scripts/download_artifacts.py` is a placeholder that
> **will not run** (its `ZENODO_RECORD` is unset and the script errors out). The
> md5 is final, so it goes live the moment the record id is filled in.

| artifact | size | md5 | for |
|---|---|---|---|
| `forma_fgrid__pf_full__test__predictions.parquet` | ~4.2 GB | `c4f0f721…` | canonical R13 **Forma** 5-seed mixture forecast (point track). Pool your model against it to reproduce **Panels A / B**, or rebuild the mask from it. |

It holds only **model outputs — no firm-level values**. This release covers the
**point track (Panels A and B)**; the density track (Panel C — exact mixture
NLL/CRPS) needs the five per-seed forecasts and is out of scope here.

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
