# ProForma-20Q

**A public benchmark for joint probabilistic forecasting of complete quarterly
financial statements at horizons 1–20 quarters.**

ProForma-20Q defines a *task*, not a model. Given your own WRDS credentials, one
command rebuilds the exact data environment and evaluation used in the companion
ICAIF '26 paper; you then train *your* model and score a forecast file with our
CLI. It is usable by someone who has never heard of the Forma model — build the
data, train, submit.

> **No firm-level data values are distributed here.** The repository contains
> code, configs and checksums, plus three Compustat-*derived* artifacts that
> carry no firm's reported figures: the published canonical regularization
> statistics (per-`(feature, quarter)` `mu`/`sigma`/`k` moments, in
> `src/proforma20q/reference/`), the Full-sample coverage bitmap, and its
> canonical row index — a `(firm, quarter)` membership list — both in
> `artifacts/`. WRDS / Compustat / CRSP credentials and license are entirely
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
# from the ROOT of a download or clone of this repository -- note the review
# mirror's zip extracts its contents directly, without a wrapper directory:
pip install -e '.[wrds,dev]'        # scoring + baselines + WRDS client + tests
```

(The quotes matter under zsh, macOS's default shell, which otherwise globs the
brackets.) Use `pip install -e '.[wrds]'` if you only need the data build, or
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

## Before you start: data access and preconditions

ProForma-20Q is a **protocol, not a dataset**. Every artifact is rebuilt from
your own Compustat licence, so the benchmark cannot be run at all without the
access below.

| source | used for | required? |
|---|---|---|
| WRDS account | all data access, via the [`wrds`](https://pypi.org/project/wrds/) client | **yes** |
| Compustat Fundamentals Quarterly (`comp.fundq`) | the 78 statement items | **yes** |
| Compustat `comp.co_industry` | SIC/NAICS → FF48 industry dummies, and the financial-sector exclusion | **yes** |
| CRSP `crsp.ccmxpf_lnkhist` (CCM link table) | **part of the sample definition** — trims linked firms to their link windows; see [The CRSP link filter](#the-crsp-link-filter) | **yes** |

> **CRSP is not optional.** The link merge trims *linked* firms to their link
> windows, which is part of the published sample definition (firms with no CCM
> link are retained in full — see [The CRSP link filter](#the-crsp-link-filter)).
> Skipping it yields a larger, systematically different universe whose scores
> are **not comparable** to the leaderboard.

### Step 0 — authenticate to WRDS

Do this first, and verify it, before running anything else.

**1. Install the client:** `pip install -e '.[wrds]'`

**2. Create a `pgpass` file** so the build is non-interactive. The path is
platform-specific — the POSIX one is not the only one:

| platform | path |
|---|---|
| Linux / macOS | `~/.pgpass` (must be `chmod 600`) |
| **Windows** | `%APPDATA%\postgresql\pgpass.conf` |

One line, five colon-delimited fields:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<your_wrds_username>:<your_wrds_password>
```

**3. ⚠ Expect Duo two-factor — even with a valid `pgpass` file.** A `pgpass`
supplies your *password*; it does not satisfy WRDS's second factor. Connecting
may send a **Duo push to your phone that you must approve**.

> **Do not retry a failed connection in a loop.** Repeated authentication
> attempts without a Duo response cause **WRDS to deactivate the account**, which
> takes a support ticket to undo. If a connection fails, stop and fix the cause.
>
> Duo may *not* prompt on every connection — device-trust windows mean an
> authentication can succeed silently now and require a push later, from a new
> machine or IP. **One silent success does not mean 2FA is not enforced.**

`proforma20q download` makes exactly one connection attempt per run and never
retries. If you are driving this repo programmatically, read
[AGENTS.md](AGENTS.md) first.

**4. Verify once:**

```bash
python -c "import wrds; db = wrds.Connection(wrds_username='<user>'); print(db.raw_sql('select count(*) from comp.fundq limit 1')); db.close()"
```

### What you can do *without* credentials

Works: `pip install -e '.[dev]'` and the full test suite (all synthetic, no WRDS);
`proforma20q validate <file>`; `proforma20q evaluate <file> --truth
examples/example_truth.parquet`; reading `task.yaml` / `feature_sets.yaml`;
verifying `artifacts/full_sample_mask_bits.npy` against its manifest.

Requires credentials, no workaround: `proforma20q build` and therefore every
real artifact, any leaderboard number, `report-drift`, and `evaluate
--sample-mask` against the Full-sample mask.

---

## Quickstart

### 1. Build the data (needs WRDS)

```bash
proforma20q build --wrds-user <your_wrds_user>
# -> authenticates ONCE (see Step 0; expect a Duo push)
# -> downloads Compustat+CRSP, processes both views, verifies checksums
proforma20q report-drift
# -> the PASS/FAIL vintage-drift verdict (see "Reproducing the canonical build")
```

The build ends with a **bit-exact md5 verification**, and on any fresh WRDS
pull it will print `mismatch` for every artifact and `ALL MATCH: False`. That
is expected — Compustat is revised, so bit-exact equality with the canonical
snapshot is not achievable ([details](#reproducing-the-canonical-build)). The
check with a real verdict is the second command: `report-drift` compares
per-column distribution statistics against the published canonical ones and
returns PASS/FAIL. (`build --report-drift` runs it in one step.)

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

### Which reg stats: the flag that changes your ground truth

`--reg-stats` selects the statistics that define the **regularized target
space**, i.e. the ground truth you are scored against. It is not a tuning knob:

| value | meaning |
|---|---|
| `canonical` **(default)** | pin the published R13 statistics shipped in `src/proforma20q/reference/`. Your targets are the leaderboard's targets. |
| `estimate` | re-estimate `mu`/`sigma`/`k` from *your* panel's train split. A different ground truth — scores are **not comparable** to anything published. |
| *a path* | pin an explicit `regularization_stats__*.parquet`. |

Building twice from an identical panel and toggling only this flag gave **0.0%
of target cells identical** and a mean |Δz| of 0.83 in a space clamped to
|z| ≤ 6, plus a 6-row change in the train split. (That magnitude was measured on
a synthetic panel whose scale distribution is unlike Compustat, so 0.83
overstates the real gap — the mechanism and the row-count change are real
regardless.) Two researchers both "following the README" would otherwise score
against different truth and produce non-comparable numbers, which is why the
default is pinned and the build prints which space it used.

### The CRSP link filter

`download` merges the CRSP-Compustat link table (`crsp.ccmxpf_lnkhist`,
`linktype ∈ {LU, LC}`, `linkprim ∈ {P, C}`). **This is part of the sample
definition**, declared in
[`task.yaml → universe.crsp_link`](src/proforma20q/configs/task.yaml), and its
outcome is a clean two-regime split:

- **Firms with a CCM link are trimmed to their link windows** — firm-quarters
  whose `datadate` falls outside every valid window are dropped.
- **Firms with no CCM link at all are retained in full**, with NaN `permno`.
  This is not a corner case: on the canonical 1970–2024 pull, 16,418 of 43,335
  firms — **495,440 firm-quarters, 29.0% of the panel** — have no link, a share
  that grows from 3.8% in the 1970s to **50.0% in the 2020s**. They are largely
  Canadian-incorporated filers plus smaller US firms (median `atq` 57.4 vs
  259.3 for linked). Carried into the shipped R13 splits: 12.5% of train,
  30.3% of val, and **34.4% of test** firm-quarters are unlinked.

A CRSP-listed universe is therefore **not** what this filter produces. What
CRSP entitlement buys you is the ability to reproduce the *trimming of linked
firms* exactly; membership itself never requires a CRSP listing. Skipping the
merge still yields a different, larger universe than the published one (the
out-of-window quarters of linked firms come back), so it is not optional
either.

The column it attaches, `permno`, is **never read downstream**. The window
trim is the point, and it is kept for comparability with the published sample.
(The output file is named `compustat_with_permno.parquet` after the column that
does not matter rather than the filter that does; the name is retained for
compatibility with existing pipelines.)

### If you train on the tuple view: the id maps

The tuple view stores `firm_id`, `account_id` and `industry_id` as **integers**,
while a submission needs the gvkey string (`"001045"`) and the pf_full item name
(`"niq"`). A build therefore also writes the three dictionaries that bridge them,
next to the artifacts:

```
firm_id_map__<suffix>.csv        account_id_map__<suffix>.csv
industry_id_map__<suffix>.csv
```

Read them with `proforma20q.build.read_id_maps(processed_dir, suffix)` rather
than a bare `read_csv`: `firm_id` is a **zero-padded** gvkey, and CSV has no
types, so a default read turns `"001045"` into the integer `1045` — which then
matches nothing in the truth file, silently.

**The account map has 79 entries, not 78.** The tuple view carries the size
deflator `scale` as an account (id **60**, between `revtq` = 59 and
`seqq` = 61) so a tuple-trained model can denormalize; the 18 items sorting
after `"scale"` therefore sit one id higher than a bare-78-item enumeration
would put them. Submissions still use only the 78 pf_full item *names* —
`scale` is never a forecast target.

These files also pin the ordering rule (ids are assigned by `sorted()`), which
matters because account and industry ids are embedding indices: a build whose
ordering differs from the one a checkpoint was trained under permutes its
embeddings without any error. **The canonical account and industry maps
therefore ship in this repository** —
[`src/proforma20q/reference/account_id_map__<suffix>.csv`](src/proforma20q/reference)
(79 rows) and `industry_id_map__<suffix>.csv` (49 rows). Both are derived
entirely from the in-repo task config and FF48 table (no WRDS data), so they
are exact for every vintage, and `report-drift` checks a build's maps against
them (any difference is a FAIL: it is precisely the silent-permutation risk
these files exist to prevent). The **firm map is deliberately not pinned**: its
gvkey universe drifts with the Compustat vintage (canonical: 41,595 firms; a
7-week-newer pull measured 41,601), so `report-drift` instead checks the
ordering rule (ids `0..n-1` by sorted gvkey) and that the firm count is within
1% of canonical — and firm ids must always be translated through the gvkey
strings, never assumed positionally comparable across builds.

### A third of the sample has no industry

FF48 comes from each firm's modal `sich`. On the canonical build **18,648 of
41,595 firms (44.8%)** map to `unknown`, so **494,970 rows (32.3%)** carry
all-zero `indff48_*` dummies. That is defensible — `unknown` is the dropped
reference level, so those rows sit at the intercept — but the industry block is
identically zero for a third of the data, which is worth knowing before you
build a model around it. Relatedly, the financial-sector exclusion keeps rows
with a *missing* `sich` (`missing != financial`), so ~7.8% of the final sample is
financial by NAICS. Both behaviours match the upstream research pipeline and are
documented in `task.yaml`.

### 2. Reproduce the reference baselines

Do this before trusting your own model's score — it is how you find out whether
your build reproduces the published numbers. **Run the two cheap ones first**;
the linear family takes the better part of a day (see
[Reference baselines](#reference-baselines)).

```bash
proforma20q baselines --which naive,fade      # ~15 min, ~8.5 GB peak
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
| `naive`      | **seasonal random walk** — forecast for horizon *h* is the newest lagged level a whole number of years before the forecast quarter (`level_{(4−h mod 4) mod 4}`: h=1 ← `level_3`, h=2 ← `level_2`, h=3 ← `level_1`, h=4 ← `level_0`, repeating). Same-fiscal-quarter alignment; the paper's baseline anchor. Every model worth shipping beats it. | measured with the current specs (2026-08 vintage, 32 GB machine): naive+fade together **14.5 min / 8.4 GB peak**, writing two full-coverage ~550M-row forecasts (~2.7 GB each) |
| `fade`       | AR(1) / fade-to-mean — one `(ρ_{i,h}, b_{i,h})` per **(item, horizon)** pair (1,560 closed-form OLS fits on train+val), so each item fades toward its mean at its own speed. | ″ |
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
| naive (seasonal RW anchor) | −0.0412 |
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
bit-identical to the canonical snapshot.

> **Bit-for-bit md5 equality with `checksums.json` is not achievable, and is not
> the target.** Three independent reasons, none removable by pinning
> dependencies: the canonical artifacts were written by a *different
> implementation* (the internal research repo) that agrees with this one to
> float32 epsilon rather than exactly; that implementation emits rows in a
> different order — `(year, firm, quarter)` vs `(firm, quarter)` here — and both
> file md5s and per-column hashes are position-sensitive; and Compustat is
> revised. Earlier releases presented md5 equality as the primary check. It never
> could have passed. **The supported check is the drift report below.**

1. **Pin the environment.** For a rebuild against pinned dependencies:
   ```bash
   pip install -r requirements-lock.txt
   ```
   It pins the **direct** dependencies (no hashes; transitive deps float), so two
   people on the same Compustat vintage get the same numbers. It is *not* the
   environment that produced the canonical artifacts — see the file's own header.
2. **Pin the query.** The exact WRDS SQL lives in `src/proforma20q/download.py`;
   the sample window and filters are in `src/proforma20q/configs/task.yaml`.
3. **Pin the target space.** Build with `--reg-stats canonical` so your targets
   are normalized against the published R13 statistics rather than re-estimated
   from your own vintage — see [Which reg-stats?](#which-reg-stats-the-flag-that-changes-your-ground-truth).
4. **Publish statistics, not data.** `src/proforma20q/checksums.json` holds file
   md5s plus, per tabular artifact, the row/column counts and a **per-column
   distribution summary** (coverage, mean, sd, p05/p50/p95 of the regularized
   values). Six aggregate scalars over a 600,000-row column reveal nothing
   firm-level — and unlike a hash, they are *comparable*.
5. **Check drift, with a verdict.**
   ```bash
   proforma20q report-drift                       # vs the published statistics
   proforma20q report-drift --reference <dir>     # vs a build you already trust
   ```
   For each artifact it reports the row-count delta and the worst per-column
   move in mean, sd and coverage, then returns **PASS/FAIL** against documented
   thresholds and **exits non-zero on FAIL**. It also checks the build's
   [id maps](#if-you-train-on-the-tuple-view-the-id-maps) against the pinned
   canonical reference: account/industry orderings must match exactly
   (embedding permutation is a FAIL); the firm map is checked for the ordering
   rule and a bounded count delta.

   | threshold | value |
   |---|---|
   | split row count, relative | ≤ 1% |
   | per column, \|Δ mean\| (z units) | ≤ 0.05 |
   | per column, \|Δ sd\| (z units) | ≤ 0.05 |
   | per column, \|Δ coverage\| (fraction finite) | ≤ 0.02 |

   Calibration, measured: the same task built by this package versus the internal
   builder off the same panel gives **0 of 2,497 columns out of tolerance**
   (worst \|Δ mean\| 1e-06, row delta 0); a build off a *synthetic* panel gives
   **290–317 of 321 columns out of tolerance** (worst \|Δ mean\| 1.39, rows −99%).
   The thresholds sit between those by five orders of magnitude. A genuine
   7-week vintage difference moved split row counts by 0.01% / 0.07% / 0.24%,
   comfortably inside the 1% row bound.

   > The **previous** metric — the fraction of per-column *hashes* that differ —
   > read **100.0% for both** of those cases, because one changed cell in 1.17M
   > flips a whole column's hash. It is retained only for an old
   > `checksums.json`, is labelled as not a drift measure, and never decides
   > pass/fail.

**Definition of done for a reproduction:** a fresh machine + WRDS login →
`build --reg-stats canonical` completes → `report-drift` returns **PASS** (exit
0) → `evaluate` on the shipped baseline forecasts reproduces the published
naive / fade / ElasticNet numbers on the Full-sample mask.

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

The mask comes in two interchangeable forms, and **which one you need depends on
your vintage**:

| form | matches by | survives vintage drift? |
|---|---|---|
| grid-aligned bit array (`.npy`, `np.packbits` of the canonical cell order — no firm identifiers) | **position** | **not alone** — it is a bitmap over an exact 352,962-row set; pair it with the canonical row index below |
| bit array **+ `--grid-rows`** (the published row index) | **value** | **yes** — the bitmap is realigned onto your grid by `(firm, origin)` |
| keys table (`firm, target, origin, horizon`) | **value** | **yes** — but building it needs the canonical `tabular_test`, which is not published |

**The prebuilt grid-aligned mask ships in this repository** at
[`artifacts/full_sample_mask_bits.npy`](artifacts/full_sample_mask_bits.npy)
(~66 MB; md5 `a36008d8…`, pinned in
[`scripts/full_sample_mask.manifest.json`](scripts/full_sample_mask.manifest.json)),
so the command above needs no download — *if* your `tabular_test` is
row-identical to the canonical one. It will not be: Compustat is revised, so a
fresh pull drifts. **Pass the canonical row index alongside it and the
bitmap works anyway** — the index ships in this repository too
([`artifacts/full_sample_grid_rows.parquet`](artifacts/full_sample_grid_rows.parquet),
1.9 MB), so this route needs no download either:

```bash
proforma20q evaluate my_forecasts.parquet --against baselines \
    --sample-mask artifacts/full_sample_mask_bits.npy \
    --grid-rows artifacts/full_sample_grid_rows.parquet
```

`--grid-rows` names which `(firm, origin)` row each bit belongs to, so `evaluate`
translates the mask onto *your* grid by value and scores the paper's cells minus
the rows your vintage lacks — reporting exactly how many that is. **This is the
route that keeps "score me on Forma's cells" a 67 MB proposition rather than a
3.7 GB one**: you never need the Forma forecast to define the sample.

The index cannot be derived from the published forecast, which is why it ships:
23,970 canonical rows (6.8%) carry no forecast at all — they contribute no mask
cells, yet still occupy grid positions the bitmap counts through.

> **Measured, on the canonical mask against a real WRDS rebuild** (a later
> vintage: 1,330 canonical rows gone, 474 new ones, net −856 = 0.24% drift):
> **326,401,062 of the 327,244,429 canonical cells realign — 99.74%.** Scoring
> the canonical Forma forecast through this route lands on 326,279,721 cells and
> gives R² **0.289232** / MAE **0.408311**, against **0.289172** / **0.408494**
> on the canonical build itself. Sixth-decimal moves — well inside the precision
> the paper reports, and far smaller than the ~5% sample difference you would
> incur by dropping the mask and scoring on your own model's coverage instead.

The keys form remains available and needs only the bit array plus the canonical
`tabular_test` — but that artifact is never published, so `--grid-rows` is the
supported route for outside users. `evaluate` names it in the error message if
you hand it a bit array that does not fit your grid. Because the reference
model's coverage is the binding constraint, the Full mask is exactly *its
finite-prediction cells ∩ truth*; `scripts/build_full_sample_mask.py` also
rebuilds and verifies it (`--expect 327244429`) offline from the Forma forecast,
reproducing the same md5.

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

**No firm-level WRDS-derived *values* are distributed** — you rebuild the
tabular/tuple artifacts yourself from your own Compustat licence
(`proforma20q build`). Three Compustat-derived artifacts do ship — two
*aggregates* and a *coverage index* — because none of them can reveal any
firm's reported figures and all are needed to reproduce the paper:

| artifact | what it is | why it is safe |
|---|---|---|
| `src/proforma20q/reference/regularization_stats__*.parquet` | per-`(feature, quarter)` `mu` / `sigma` / `k` | cross-sectional moments over thousands of firms; used by `--reg-stats canonical` to pin the target space |
| [`artifacts/full_sample_mask_bits.npy`](artifacts/full_sample_mask_bits.npy) | a coverage bitmap | one bit per grid cell, no identifiers and no values |
| [`artifacts/full_sample_grid_rows.parquet`](artifacts/full_sample_grid_rows.parquet) | the canonical row index the mask is a bitmap over | says only *which* `(firm, quarter)` pairs are in the test split — membership, never a reported figure |

The **Full-sample mask and its row index ship in this repository** — a
coverage bitmap plus the row labels it counts through, no firm-level values:

| artifact | size | md5 | for |
|---|---|---|---|
| [`artifacts/full_sample_mask_bits.npy`](artifacts/full_sample_mask_bits.npy) | 66 MB | `a36008d8…` | the 327,244,429-cell Full-sample mask (grid-aligned packbits, no firm ids); pass to `evaluate --sample-mask`. |
| [`artifacts/full_sample_grid_rows.parquet`](artifacts/full_sample_grid_rows.parquet) | 1.9 MB | `adbc2ae6…` | the canonical **row index** (`grid_row, firm, origin`) the mask is a bitmap over. Pass as `evaluate --grid-rows` to apply the mask to a vintage-drifted rebuild — see [the mask section](#reproduce-the-papers-pooled-columns). A byte-identical copy sits in the archival deposit; **this in-repo copy is the one to use.** |

The remaining artifacts are deposited at
**[doi:10.5281/zenodo.21269003](https://doi.org/10.5281/zenodo.21269003)**
(CC BY 4.0, ~21.6 GB total). Fetch and verify them with:

```bash
python scripts/download_artifacts.py --out data/artifacts
python scripts/download_artifacts.py --only full_sample_grid_rows.parquet   # or a subset
```

Every file is md5-checked against the pins below before it is accepted, and a
download is staged under `.part` until it verifies — a truncated or substituted
file fails loudly instead of scoring silently against the wrong bytes. Exhibit
labels refer to the paper's Table 1 (Panel A = squared-error track, Panel B =
absolute-error track):

| artifact | size | md5 | for |
|---|---|---|---|
| `forma_fgrid__pf_full__test__predictions.parquet` | 3.7 GB | `1820fcc9…` | canonical R13 **Forma** 5-seed Gaussian mixture (squared-error track). Pool your model against it to reproduce the **Panel A** Full column, or rebuild the mask from it. |
| `ffnn_linear_b50__pf_full__test__predictions.parquet` | 4.4 GB | `e419c833…` | **FFNN (linear)** 5-seed mixture — Panel A comparator row. |
| `ffnn_large_b50__pf_full__test__predictions.parquet` | 4.5 GB | `915779a3…` | **FFNN (large)** 5-seed mixture — Panel A comparator row. |
| `forma_lap05_fgrid__pf_full__test__predictions.parquet` | 7.4 GB | `1e8b0415…` | canonical R13 **Forma** Laplace mixture (absolute-error track) — the **Panel B** Full column. |
| `forma_lap05_fgrid__pf_full__test__predictions.nll.json` | 33 B | `a3d8659a…` | the Laplace file's **family sidecar**. Keep it next to the parquet (the evaluator reads `{stem}.nll.json`); without it the file is **silently scored as Gaussian**. |

The deposit also carries byte-identical copies of the in-repo mask and row
index (same md5s as the table above), so the archival record is complete on its
own — but nothing in this README requires downloading them: the in-repo copies
are canonical.

> **Joining these files to your own build:** `origin`/`quarter` in the forecasts
> is a quarter-*end* timestamp at **microsecond** precision, while a canonical
> `tabular_test` stores nanoseconds — `…23:59:59.999999` vs `…23:59:59.999999999`,
> which are **not equal** as raw timestamps. `proforma20q` is immune (it
> canonicalizes every origin to a calendar quarter), but a hand-written
> `pd.merge` on the raw column silently returns nothing. Compare on
> `pd.to_datetime(s).dt.to_period("Q")`. `full_sample_grid_rows.parquet` sidesteps
> this by storing `origin` as a plain ISO date.

They hold only **model outputs — no firm-level values**. This release covers
the **point track (Panels A and B)**; the density track (Panel C — exact
mixture NLL/CRPS) needs the five per-seed forecasts and is out of scope here
([#3](https://github.com/ANONYMIZED/proforma-20q/issues/3) tracks it).

### Documentation

| artifact | what it is |
|---|---|
| [`docs/release_documentation.pdf`](docs/release_documentation.pdf) | Detailed technical documentation in three parts: **A** the data pipeline (sample formation, splits, regularization, the 78-item universe, availability, and the accounting identities), **B** model training/implementation detail and every competitor's exact specification, **C** the LLM benchmark protocol with the prompts reproduced in full. |
| [`docs/prompts/`](docs/prompts/) | The byte-exact system prompts for both elicitation arms, with published SHA-256 anchors ([`docs/README.md`](docs/README.md)). Part C's typeset copies are ASCII transliterations; these files are the originals. |

> **This repository holds the canonical `release_documentation.pdf`**
> (`sha256 55fa6f76dffaca29b1027a8438d5c09e26dc182d89790f50ef9ec5803a46fa5e`).
> A byte-identical copy also ships in the companion model repository, so that
> anyone holding only that repository still has Parts B and C, which describe
> its code. If the two digests ever differ, this copy wins — regenerate both
> together.

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
firm-level data is distributed** (see [Data & artifacts](#data--artifacts)); you
are responsible for your WRDS/Compustat/CRSP license.
The FF48 industry classification derives from the Kenneth R. French Data Library.

If you use ProForma-20Q, please cite it — see [CITATION.cff](CITATION.cff).
