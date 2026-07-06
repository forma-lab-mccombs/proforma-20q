"""Checksums and vintage-drift reporting for built artifacts.

Constraint (README / brief Sec 3.1): no WRDS-derived *values* are distributed --
so we publish only hashes plus coarse aggregate shape, and verify a user's
rebuild against them.

* ``md5`` of each artifact file -> exact bit-for-bit reproduction check.
* per-column md5 of the regularized target/level columns (values rounded before
  hashing) -> a coarse *fraction of columns that diverge*, used by
  ``report-drift`` since a fresh WRDS pull is never bit-identical (Compustat is
  revised). It flags WHICH parts moved without revealing any value.
* per-tabular ``n_rows`` / ``n_cols`` -> aggregate scalars (no firm-level data)
  that let ``report-drift`` report the row-count delta of a diverging vintage.

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

# Artifacts a full build produces (relative to the processed dir), by role.
ARTIFACT_SPLITS = ("train", "val", "test")


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
    """Per-column md5 of the rounded regularized values (NaN-stable)."""
    out: dict[str, str] = {}
    for c in _regularized_columns(df):
        arr = np.round(df[c].to_numpy(dtype="float64"), decimals)
        out[c] = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
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
    them. Only hashes are written -- never any WRDS-derived value.
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


def report_drift(processed_dir, suffix: str, published: dict | None = None) -> dict:
    """Vintage-drift report: per-tabular-artifact fraction of columns whose hash
    differs from the canonical build (a coarse divergence measure that does not
    require -- or reveal -- the canonical data).

    Returns ``{artifact: {"n_cols", "n_diff", "frac_diff", "file_md5_match",
    "row_count_delta"}}``.
    """
    published = published or load_published_checksums()
    if published.get("_status") != "populated":
        return {"_note": "published checksums are unpopulated; nothing to compare against"}
    pub = published.get("artifacts", {})
    report: dict = {}
    for name, path in artifact_paths(processed_dir, suffix).items():
        pentry = pub.get(name)
        if pentry is None:
            report[name] = {"note": "no published entry"}
            continue
        entry: dict = {"file_md5_match": hash_file(path) == pentry["md5"]}
        if name.startswith("tabular_") and "column_md5" in pentry:
            df = pd.read_parquet(path, engine=PARQUET_ENGINE)
            built_cols = column_hashes(df)
            pub_cols = pentry["column_md5"]
            shared = set(built_cols) & set(pub_cols)
            n_diff = sum(1 for c in shared if built_cols[c] != pub_cols[c])
            entry.update({
                "n_cols": len(shared),
                "n_diff": n_diff,
                "frac_diff": (n_diff / len(shared)) if shared else float("nan"),
                "cols_only_here": sorted(set(built_cols) - set(pub_cols))[:10],
                "cols_only_published": sorted(set(pub_cols) - set(built_cols))[:10],
                "row_count_delta": int(len(df) - pentry.get("n_rows", len(df))),
            })
        report[name] = entry
    return report
