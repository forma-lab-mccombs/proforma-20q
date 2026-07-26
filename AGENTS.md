# Operating this repository programmatically

Read [README.md](README.md) first for what the benchmark *is*. This file is the
set of rules for running it **unattended**. Each one corresponds to a real
failure that is cheap for a human to notice and expensive for an agent.

Facts about the benchmark live once, in README.md and `task.yaml`, and are linked
from here rather than restated.

---

## 1. WRDS authentication — the only rule that can cost real-world access

**On any authentication failure: stop, report the error, and ask a human.**

- **Never retry a WRDS connection programmatically.** No loops, no backoff, no
  "try once more". Repeated attempts without a Duo response cause **WRDS to
  deactivate the account**, which needs a support ticket to restore. The blast
  radius is the user's institutional access, not your run.
- **Duo 2FA applies even with a valid `pgpass` file.** A connection may block on
  a push to the user's phone. Running without a human present? Say so and stop.
- **A silent success proves nothing.** Device-trust windows mean auth can succeed
  with no prompt and require one later. Do not conclude 2FA is unenforced.
- `download()` makes exactly one connection attempt and never retries — it drives
  the `wrds` client's single-attempt path deliberately, because
  `wrds.Connection.connect()` falls back to prompting and connecting a *second*
  time. Do not replace it with a bare `wrds.Connection(...)`.
- **Open one connection and reuse it** for multi-step work. Every entry point
  takes `connection=`:

  ```python
  import wrds
  from proforma20q.download import download
  db = wrds.Connection(wrds_username=USER)      # authenticate ONCE
  try:
      download(USER, connection=db)             # chunks internally; one auth
  finally:
      db.close()
  ```

  `download()` already pulls year-by-year and caches each chunk, so an
  interrupted pull resumes without re-authenticating. Do **not** write your own
  per-year loop that calls `download()` without `connection=` — that is 55
  separate authentications.
- Never read, log, echo, or copy the contents of the `pgpass` file.

## 2. Never edit the task definition to make something pass

These files *are* the benchmark. Changing any of them silently defines a
different benchmark and invalidates every published number:

- `src/proforma20q/checksums.json`
- `src/proforma20q/configs/task.yaml`
- `src/proforma20q/configs/feature_sets.yaml`
- `src/proforma20q/configs/ff48_sic_ranges.json`
- `src/proforma20q/reference/regularization_stats__*.parquet`
- `artifacts/full_sample_mask_bits.npy`

If checksum verification fails or `report-drift` FAILs, **that is a result to
report, not a test to fix.** Do not adjust expected values, relax a threshold, or
regenerate a pinned artifact to get a green run. A benchmark built on an edited
definition is worse than a failing one, because it looks correct.

Two specific traps:

- **`report-drift` thresholds** live in `checksums.py::DRIFT_THRESHOLDS` and are
  documented in the README. Widening them to turn a FAIL into a PASS is the same
  offence as editing `checksums.json`.
- **File md5 equality with `checksums.json` is not achievable** from this
  repository, for reasons in `requirements-lock.txt`. A mismatch there is
  expected; do not go looking for a way to "fix" it.

## 3. Resource preconditions — check before launching, not after

Measured on the canonical 1970–2024 panel, 34 GB workstation:

| step | wall | peak RSS |
|---|---|---|
| `download` (default: 1-year chunks, 82-column projection) | ~1 h | a few GB |
| `download --all-columns` in one call | — | **~100 GB; does not complete** |
| `build --which tabular` | ~15 min | **20 GB** |
| `baselines --which naive,fade` | ~17 min | 11 GB |
| `baselines --which linear` | **7–14 h** | ~10 GB |
| `baselines --which elasticnet` | **2–4 h** | ~10 GB |

- `linear` is the **most** expensive baseline, not the cheapest. Do not treat
  "reproduce the baselines" as a quick check.
- A full-coverage forecast is **549,285,360 rows**. Never assemble one as a
  single frame; use `proforma20q.schema.write_forecast_blocks`. Reading one back
  as a frame needs ~73 GB — `validate` streams it by row-group instead.
- When concatenating per-year panels yourself, take dtypes from the **Postgres
  declarations**, not from the chunks: a column that is entirely NULL in one year
  arrives as `object` and will stringify a genuinely numeric column across every
  other year. `download()` already does this.

## 4. Long-running steps look like hangs

- Run with `python -u` (or `PYTHONUNBUFFERED=1`). `build` prints progress bars;
  through a pipe, block buffering can still make them invisible.
- Do not kill a silent `build`. Poll RSS or the output directory instead.

## 5. Exit codes

| code | meaning |
|---|---|
| `0` | success (for `report-drift`: PASS) |
| `1` | authentication failure; invalid forecast file; common-sample invariant violated; **drift beyond threshold** |
| `2` | precondition missing (no build present, no raw panel, unreadable forecast) |

When checking exit status in a shell, **do not read `$?` after a pipeline** —
`cmd | tail` reports `tail`'s status. Redirect to a file, or use
`${PIPESTATUS[0]}`.

## 6. Do not commit data

`.gitignore` excludes `data/`, `results/`, and `*.parquet`. Never override it
(`git add -f`) for WRDS-derived content. Committing derived data is a licence
violation, not a style issue.

## 7. Scratch work goes outside the repository

Write intermediate panels, logs, and experiments outside the clone. Do not add
helper scripts to `scripts/` as a side effect of a task; propose them.

---

## Reporting back

State:

- the exact commands run and their exit codes;
- for any build: row counts per split, which reg-stats space was used
  (`--reg-stats canonical` vs `estimate` — they are different ground truths),
  and the `report-drift` verdict;
- for any score: `n_complete` and whether the common-sample invariant held;
- **anything you could not verify**, explicitly.

Coverage numbers are load-bearing: `evaluate` scores on the strict intersection
of every model passed, so one submission with non-finite predictions silently
shrinks *everyone's* sample. Report `n_complete`, never just the metric.
