"""Checksums and vintage-drift reporting for built artifacts.

Constraint (README / brief Sec 3.1): no WRDS-derived *values* are distributed --
so we publish only hashes plus coarse aggregate shape, and verify a user's
rebuild against them.

* ``md5`` of each artifact file -> exact bit-for-bit reproduction check.
* per-column md5 of the regularized target/level columns (values rounded before
  hashing) -> a coarse *fraction of columns that diverge*. Position-sensitive,
  so it is never a drift measure; retained for the bit-exact path.
* per-tabular ``n_rows`` / ``n_cols`` -> aggregate scalars (no firm-level data)
  that let ``report-drift`` report the row-count delta of a diverging vintage.
* per-column ``column_stats`` (coverage, mean, sd, p05/p50/p95 of the
  regularized values) -> the *comparable* statistics ``report-drift`` scores
  against :data:`DRIFT_THRESHOLDS` to render its PASS/FAIL verdict. Six
  aggregate moments over a 600k-row column reveal nothing firm-level.

The bundled ``checksums.json`` is populated by the benchmark maintainer from the
canonical build (``write_checksums``); users compare against it with
``verify_checksums`` / ``report_drift``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import PARQUET_ENGINE

CHECKSUMS_PATH = Path(__file__).resolve().parent / "checksums.json"
_HASH_DECIMALS = 6  # round regularized values before column hashing
_STAT_DECIMALS = 6  # round published per-column moments

# Artifacts a full build produces (relative to the processed dir), by role.
ARTIFACT_SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Drift pass thresholds
# ---------------------------------------------------------------------------
# A build is ACCEPTED when every artifact stays inside all four. They are stated
# here, in code, so "how much drift is too much" has one answer that a script can
# check -- the previous "definition of done" had none.
#
# Calibration. The two builds that must be distinguishable are (i) the same task
# off a Compustat vintage 7 weeks newer, whose split row counts moved by
# 0.01% / 0.07% / 0.24%, and (ii) a build off a panel that is not this task at
# all. `--reference` runs against a local reference build if you have one; the
# published values in checksums.json are the fallback.
DRIFT_THRESHOLDS = {
    "max_row_delta_frac": 0.01,   # split row count, relative
    "max_abs_mean_delta": 0.05,   # per column, in z units (space is clamped to 6)
    "max_abs_sd_delta": 0.05,     # per column, in z units
    "max_abs_coverage_delta": 0.02,   # per column, fraction of rows finite
}


def hash_file(path) -> str:
    """Streaming md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _regularized_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c.startswith("indff48_") is False
        and (("_level_" in c) or ("_yoy_" in c) or _is_target_col(c))
    )


def _is_target_col(c: str) -> bool:
    if "_t" not in c:
        return False
    tail = c.rsplit("_t", 1)[-1]
    return tail.isdigit()


def column_hashes(df: pd.DataFrame, decimals: int = _HASH_DECIMALS) -> dict[str, str]:
    """Per-column md5 of the rounded regularized values (NaN-stable).

    Retained for the bit-exact ``verify_checksums`` path. **Not** usable as a
    drift measure: one changed cell in 1.17M flips a column's hash, and the hash
    is position-sensitive, so any real rebuild reads ~100% diverged whatever the
    data says. :func:`column_stats` is what ``report-drift`` compares.
    """
    out: dict[str, str] = {}
    for c in _regularized_columns(df):
        arr = np.round(df[c].to_numpy(dtype="float64"), decimals)
        out[c] = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return out


def column_stats(df: pd.DataFrame, decimals: int = _STAT_DECIMALS) -> dict[str, dict]:
    """Per-column distribution summary of the regularized values.

    Six aggregate scalars per column -- coverage, mean, sd, and the 5/50/95
    percentiles, all in the regularized ``|z| <= 6`` space. Same privacy posture
    as the ``n_rows`` / ``n_cols`` already published: no firm-level value can be
    recovered from a moment of a 600,000-row column.

    Why this and not the hash: these are *comparable*. Two builds of the same
    task off different Compustat vintages differ in the last digits of a few
    cells, so every statistic moves by ~1e-3; a build off the wrong panel moves
    them by whole z units. A hash cannot tell those apart -- it says "different"
    to both.
    """
    out: dict[str, dict] = {}
    for c in _regularized_columns(df):
        arr = df[c].to_numpy(dtype="float64")
        finite = arr[np.isfinite(arr)]
        n = int(finite.size)
        if n == 0:
            out[c] = {"n_finite": 0, "coverage": 0.0}
            continue
        q05, q50, q95 = np.percentile(finite, [5, 50, 95])
        out[c] = {
            "n_finite": n,
            "coverage": round(float(n / len(arr)), _STAT_DECIMALS),
            "mean": round(float(finite.mean()), _STAT_DECIMALS),
            "sd": round(float(finite.std(ddof=0)), _STAT_DECIMALS),
            "p05": round(float(q05), _STAT_DECIMALS),
            "p50": round(float(q50), _STAT_DECIMALS),
            "p95": round(float(q95), _STAT_DECIMALS),
        }
    return out


def artifact_paths(processed_dir, suffix: str) -> dict[str, Path]:
    """Map artifact-name -> path for a built ``suffix`` (existing files only)."""
    processed_dir = Path(processed_dir)
    names = [f"regularization_stats__{suffix}.parquet"]
    for view in ("tabular", "tuple"):
        for split in ARTIFACT_SPLITS:
            names.append(f"{view}_{split}__{suffix}.parquet")
    return {n: processed_dir / n for n in names if (processed_dir / n).exists()}


def compute_checksums(processed_dir, suffix: str) -> dict:
    """Compute the full checksum record for a build (file md5 + column hashes for
    tabular artifacts)."""
    record: dict = {"suffix": suffix, "artifacts": {}}
    for name, path in artifact_paths(processed_dir, suffix).items():
        entry = {"md5": hash_file(path)}
        if name.startswith("tabular_"):
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
            entry["n_rows"] = int(len(df))
            entry["n_cols"] = int(df.shape[1])
            entry["column_md5"] = column_hashes(df)
            entry["column_stats"] = column_stats(df)
        record["artifacts"][name] = entry
    return record


def write_checksums(
    processed_dir,
    suffix: str,
    out_path=CHECKSUMS_PATH,
    *,
    download_date: str | None = None,
    task_version: str | None = None,
) -> Path:
    """(Maintainer) Populate ``checksums.json`` from the canonical build.

    ``download_date`` (the WRDS pull date, ISO ``YYYY-MM-DD``) and
    ``task_version`` are recorded alongside the hashes; when omitted they are
    carried forward from the existing ``checksums.json`` so a re-run preserves
    them. Only hashes and aggregate per-column statistics are written -- never
    any firm-level value.
    """
    out_path = Path(out_path)
    prior: dict = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prior = {}
    core = compute_checksums(processed_dir, suffix)
    resolved_dl = download_date if download_date is not None else prior.get("download_date")
    resolved_tv = task_version or prior.get("task_version")
    if resolved_dl is None or resolved_tv is None:
        print(f"  Warning: writing checksums with download_date={resolved_dl!r}, "
              f"task_version={resolved_tv!r} (no value passed and none in {out_path.name}).")
    record = {
        "_status": "populated",
        "task_version": resolved_tv,
        "suffix": core["suffix"],
        "download_date": resolved_dl,
        "artifacts": core["artifacts"],
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


def load_published_checksums(path=CHECKSUMS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_checksums(processed_dir, suffix: str, published: dict | None = None) -> dict:
    """Bit-exact check: compare each built artifact's md5 to the published one.

    Returns ``{artifact: {"status": ok|mismatch|missing|extra, "md5": ...}}`` plus
    an ``"all_match"`` bool.
    """
    published = published or load_published_checksums()
    if published.get("_status") != "populated":
        return {"all_match": False, "_note": "published checksums are unpopulated"}
    pub = published.get("artifacts", {})
    built = compute_checksums(processed_dir, suffix)["artifacts"]
    report: dict = {}
    all_match = True
    for name, pentry in pub.items():
        bentry = built.get(name)
        if bentry is None:
            report[name] = {"status": "missing"}
            all_match = False
        elif bentry["md5"] == pentry["md5"]:
            report[name] = {"status": "ok", "md5": bentry["md5"]}
        else:
            report[name] = {"status": "mismatch",
                            "published": pentry["md5"], "built": bentry["md5"]}
            all_match = False
    for name in built:
        if name not in pub:
            report[name] = {"status": "extra"}
    report["all_match"] = all_match
    return report


def _compare_column_stats(built: dict, ref: dict, thresholds: dict) -> dict:
    """Cell-level-ish divergence between two column-stat tables.

    Reports the worst per-column move in each dimension, plus how many columns
    exceed the threshold and which ones -- so the answer to "did my build come
    out right" is a number with a scale, not a boolean.
    """
    shared = sorted(set(built) & set(ref))
    only_here = sorted(set(built) - set(ref))
    only_ref = sorted(set(ref) - set(built))
    worst = {"mean": 0.0, "sd": 0.0, "coverage": 0.0}
    worst_col = {"mean": None, "sd": None, "coverage": None}
    offenders: list[str] = []
    deltas: dict[str, float] = {}
    for c in shared:
        b, r = built[c], ref[c]
        if not b.get("n_finite") or not r.get("n_finite"):
            # a column that is empty on one side and not the other IS the finding
            if bool(b.get("n_finite")) != bool(r.get("n_finite")):
                offenders.append(c)
            continue
        d = {
            "mean": abs(b["mean"] - r["mean"]),
            "sd": abs(b["sd"] - r["sd"]),
            "coverage": abs(b["coverage"] - r["coverage"]),
        }
        deltas[c] = d["mean"]
        for k, v in d.items():
            if v > worst[k]:
                worst[k], worst_col[k] = v, c
        if (d["mean"] > thresholds["max_abs_mean_delta"]
                or d["sd"] > thresholds["max_abs_sd_delta"]
                or d["coverage"] > thresholds["max_abs_coverage_delta"]):
            offenders.append(c)
    mean_deltas = np.array(list(deltas.values())) if deltas else np.zeros(0)
    return {
        "n_cols": len(shared),
        "n_cols_over_threshold": len(offenders),
        # A column set that does not line up is itself the finding. Reported so a
        # mismatch cannot pass as "0 columns over threshold" -- the symmetric
        # failure to the old metric, and the more dangerous direction.
        "cols_only_here": only_here[:10],
        "cols_only_published": only_ref[:10],
        "n_cols_only_here": len(only_here),
        "n_cols_only_published": len(only_ref),
        "frac_cols_over_threshold": (len(offenders) / len(shared)) if shared else float("nan"),
        "worst_abs_mean_delta": round(worst["mean"], 6),
        "worst_abs_mean_delta_col": worst_col["mean"],
        "worst_abs_sd_delta": round(worst["sd"], 6),
        "worst_abs_sd_delta_col": worst_col["sd"],
        "worst_abs_coverage_delta": round(worst["coverage"], 6),
        "worst_abs_coverage_delta_col": worst_col["coverage"],
        "median_abs_mean_delta": round(float(np.median(mean_deltas)), 8) if mean_deltas.size else 0.0,
        "cols_over_threshold": sorted(offenders)[:10],
    }


def reference_record(reference_dir, suffix: str) -> dict:
    """A published-checksums-shaped record computed from a local reference build.

    Lets ``report-drift --reference`` compare against a build you have on disk
    rather than against the published aggregates.
    """
    core = compute_checksums(reference_dir, suffix)
    return {"_status": "populated", "suffix": core["suffix"],
            "artifacts": core["artifacts"], "_source": str(reference_dir)}


def report_drift(processed_dir, suffix: str, published: dict | None = None,
                 *, thresholds: dict | None = None) -> dict:
    """Vintage-drift report with a verdict.

    Compares the local build's per-column distribution summary against the
    published one (or a reference build's, via :func:`reference_record`) and
    decides PASS/FAIL against :data:`DRIFT_THRESHOLDS`. Neither side needs the
    other's data: the comparison is between aggregate moments.

    Iterates the *published* artifact list (like :func:`verify_checksums`), so a
    published artifact with no local file surfaces as ``{"status": "missing"}``
    rather than being silently absent -- a machine with no build must not read
    as "no drift". Local files with no published entry surface as
    ``{"status": "extra"}``.

    Returns ``{artifact: {...}}`` plus ``"_pass"`` (bool or None when there is
    nothing comparable) and ``"_thresholds"``.
    """
    published = published or load_published_checksums()
    if published.get("_status") != "populated":
        return {"_note": "published checksums are unpopulated; nothing to compare against",
                "_pass": None}
    thresholds = dict(DRIFT_THRESHOLDS, **(thresholds or {}))
    pub_suffix = published.get("suffix")
    if pub_suffix and pub_suffix != suffix:
        # Comparing a differently-tagged build against the published artifacts
        # would report every one of them "missing" and FAIL, which says nothing
        # about drift.
        return {"_note": (f"this build is tagged '{suffix}' but the reference is "
                          f"'{pub_suffix}'; there is nothing to compare. Rebuild with "
                          f"the canonical tag (omit --tag) to check drift."),
                "_pass": None}
    pub = published.get("artifacts", {})
    present = artifact_paths(processed_dir, suffix)  # existing files only
    report: dict = {"_thresholds": thresholds}
    verdicts: list[bool] = []
    have_stats = False
    for name, pentry in pub.items():
        path = present.get(name)
        if path is None:
            report[name] = {"status": "missing"}
            verdicts.append(False)
            continue
        entry: dict = {"file_md5_match": hash_file(path) == pentry["md5"]}
        if name.startswith("tabular_"):
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
            n_pub = pentry.get("n_rows", len(df))
            row_delta = int(len(df) - n_pub)
            entry["row_count_delta"] = row_delta
            entry["row_count_delta_frac"] = (abs(row_delta) / n_pub) if n_pub else float("nan")

            if "column_stats" in pentry:
                have_stats = True
                entry.update(_compare_column_stats(
                    column_stats(df), pentry["column_stats"], thresholds))
                # A comparison over ZERO shared columns is not a pass -- it is a
                # non-comparison, and passing it on similar row counts alone
                # would certify a build nothing was actually checked against.
                # Likewise any column present on one side only.
                ok = (entry["n_cols"] > 0
                      and entry["n_cols_only_here"] == 0
                      and entry["n_cols_only_published"] == 0
                      and entry["row_count_delta_frac"] <= thresholds["max_row_delta_frac"]
                      and entry["n_cols_over_threshold"] == 0)
                entry["pass"] = bool(ok)
                verdicts.append(bool(ok))
            elif "column_md5" in pentry:
                # Legacy metric, kept only so an old checksums.json still says
                # something. It is structurally ~100% for any real rebuild: one
                # changed cell flips a whole column's hash, and the hash is
                # position-sensitive while row ORDER is not part of the task
                # definition. Never a pass/fail input.
                built_cols = column_hashes(df)
                pub_cols = pentry["column_md5"]
                shared = set(built_cols) & set(pub_cols)
                n_diff = sum(1 for c in shared if built_cols[c] != pub_cols[c])
                entry.update({
                    "n_cols": len(shared),
                    "n_diff": n_diff,
                    "frac_diff": (n_diff / len(shared)) if shared else float("nan"),
                    "hash_metric_only": True,
                })
        report[name] = entry
    for name in present:
        if name not in pub:
            report[name] = {"status": "extra"}

    # Id maps: the tuple view's embedding orderings, checked against the pinned
    # canonical reference (or the reference build's own maps). None when the
    # build wrote no id maps (a tabular-only build) -- that is not a failure,
    # the maps only exist to align tuple-view integer ids.
    from .build import verify_id_maps
    id_rep = verify_id_maps(processed_dir, suffix,
                            reference_dir=published.get("_source"))
    if id_rep is not None:
        report["_id_maps"] = id_rep

    if not have_stats:
        report["_note"] = (
            "the published checksums carry only per-column HASHES, which cannot "
            "measure drift (one changed cell flips a column, and the hash is "
            "position-sensitive). Only the row-count delta below is meaningful. "
            "Compare against a reference build with `--reference <dir>`, or ask "
            "the maintainer to republish checksums.json with column_stats.")
    id_ok = id_rep["_ok"] if id_rep is not None else True
    if (verdicts and not all(verdicts)) or not id_ok:
        # a missing artifact or a permuted/missing id map is a failure whether
        # or not comparable statistics were published
        report["_pass"] = False
    elif not have_stats:
        report["_pass"] = None           # nothing comparable was published
    else:
        report["_pass"] = bool(verdicts)
    return report
