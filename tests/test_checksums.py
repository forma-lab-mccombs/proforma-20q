"""Round-trip tests for checksum population / verification / drift.

No WRDS data: synthetic 'processed' artifacts named per the suffix convention
exercise write_checksums -> verify_checksums -> report_drift, and lock in that
download_date / task_version survive population (issue #1's documented workflow).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from proforma20q.checksums import (
    report_drift,
    verify_checksums,
    write_checksums,
)
from proforma20q.schema import PARQUET_ENGINE

SUFFIX = "pf_full__r13_node_optionD_indfe_val8"


def _tabular_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 40
    return pd.DataFrame({
        "firm_id": [f"{i:05d}" for i in range(n)],
        "quarter": pd.period_range("2010Q1", periods=n, freq="Q").to_timestamp(how="end"),
        "niq_level_0": rng.standard_normal(n),
        "niq_yoy_0": rng.standard_normal(n),
        "niq_t1": rng.standard_normal(n),
        "revtq_level_0": rng.standard_normal(n),
        "revtq_t4": rng.standard_normal(n),
        "indff48_0": rng.integers(0, 2, n).astype("float32"),
    })


def _make_processed(dirpath) -> None:
    """Write a minimal but well-named build into ``dirpath``."""
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        _tabular_frame(seed).to_parquet(
            dirpath / f"tabular_{split}__{SUFFIX}.parquet", engine=PARQUET_ENGINE, index=False)
    pd.DataFrame({"quarter": pd.period_range("2010Q1", periods=3, freq="Q").to_timestamp(how="end"),
                  "mu": [0.0, 0.1, 0.2], "sigma": [1.0, 1.0, 1.0],
                  "feature": ["niq", "revtq", "niq"], "k": [1.0, 1.0, 1.0]}).to_parquet(
        dirpath / f"regularization_stats__{SUFFIX}.parquet", engine=PARQUET_ENGINE, index=False)


def test_write_verify_roundtrip(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    _make_processed(proc)
    out = tmp_path / "checksums.json"

    write_checksums(proc, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["_status"] == "populated"
    assert rec["download_date"] == "2026-07-02"
    assert rec["task_version"] == "r13"
    assert rec["suffix"] == SUFFIX
    assert rec["artifacts"], "artifacts should be populated"
    assert any(name.startswith("tabular_") and "column_md5" in e
               for name, e in rec["artifacts"].items())

    published = json.loads(out.read_text(encoding="utf-8"))
    rep = verify_checksums(proc, SUFFIX, published=published)
    assert rep["all_match"] is True

    drift = report_drift(proc, SUFFIX, published=published)
    for name, r in drift.items():
        if "frac_diff" in r:
            assert r["frac_diff"] == 0.0
            assert r["file_md5_match"] is True


def test_download_date_preserved_on_rerun(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    _make_processed(proc)
    out = tmp_path / "checksums.json"

    write_checksums(proc, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    write_checksums(proc, SUFFIX, out_path=out)  # no date/version passed
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["download_date"] == "2026-07-02"
    assert rec["task_version"] == "r13"


def test_drift_detects_changed_values(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    _make_processed(proc)
    out = tmp_path / "checksums.json"
    write_checksums(proc, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    published = json.loads(out.read_text(encoding="utf-8"))

    # Perturb one column of the test tabular artifact -> a vintage divergence.
    df = pd.read_parquet(proc / f"tabular_test__{SUFFIX}.parquet", engine=PARQUET_ENGINE)
    df["niq_level_0"] = df["niq_level_0"] + 1.0
    df.to_parquet(proc / f"tabular_test__{SUFFIX}.parquet", engine=PARQUET_ENGINE, index=False)

    assert verify_checksums(proc, SUFFIX, published=published)["all_match"] is False
    drift = report_drift(proc, SUFFIX, published=published)
    test_entry = drift[f"tabular_test__{SUFFIX}.parquet"]
    assert test_entry["n_diff"] >= 1
    assert test_entry["frac_diff"] > 0.0
    assert test_entry["file_md5_match"] is False
