"""Round-trip tests for checksum population / verification / drift.

No WRDS data: synthetic 'processed' artifacts named per the suffix convention
exercise write_checksums -> verify_checksums -> report_drift, and lock in that
download_date / task_version survive population (issue #1's documented workflow).
"""
from __future__ import annotations

import json
import shutil

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
    assert drift["_pass"] is True
    for name, r in drift.items():
        if name.startswith("_"):
            continue
        if "n_cols_over_threshold" in r:
            assert r["n_cols_over_threshold"] == 0
            assert r["worst_abs_mean_delta"] == 0.0
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


def test_drift_empty_processed_reports_all_missing(tmp_path):
    """No build at all: report_drift must surface every published artifact as
    'missing' rather than returning an empty (green-looking) report."""
    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    out = tmp_path / "checksums.json"
    write_checksums(ref, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    published = json.loads(out.read_text(encoding="utf-8"))

    empty = tmp_path / "processed"
    empty.mkdir()
    drift = report_drift(empty, SUFFIX, published=published)
    artifacts = {k: v for k, v in drift.items() if not k.startswith("_")}
    assert artifacts, "empty build must not yield an empty drift report"
    assert set(artifacts) == set(published["artifacts"])
    assert all(r.get("status") == "missing" for r in artifacts.values())
    assert drift["_pass"] is False


def test_drift_partial_build_mixes_missing_and_reported(tmp_path):
    """A partial build reports the present artifact and marks the rest missing."""
    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    out = tmp_path / "checksums.json"
    write_checksums(ref, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    published = json.loads(out.read_text(encoding="utf-8"))

    proc = tmp_path / "processed"
    proc.mkdir()
    only = f"tabular_test__{SUFFIX}.parquet"
    shutil.copy(ref / only, proc / only)

    drift = report_drift(proc, SUFFIX, published=published)
    assert drift[only]["n_cols_over_threshold"] == 0 and drift[only]["file_md5_match"] is True
    missing = [n for n, r in drift.items()
               if not n.startswith("_") and r.get("status") == "missing"]
    assert missing and only not in missing


def test_report_drift_cli_exits_2_on_empty_machine(tmp_path, capsys):
    """`proforma20q report-drift` on a machine with no build exits 2 with an
    explicit message (consistent with build/evaluate missing-data behavior),
    using the shipped (populated) canonical checksums."""
    from proforma20q.cli import main

    empty = tmp_path / "processed"
    empty.mkdir()
    rc = main(["report-drift", "--processed", str(empty)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no built artifacts found" in err
    assert "proforma20q build" in err


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
    assert test_entry["n_cols_over_threshold"] >= 1
    assert test_entry["worst_abs_mean_delta"] > 0.5      # a whole z unit moved
    assert test_entry["worst_abs_mean_delta_col"] == "niq_level_0"
    assert test_entry["pass"] is False
    assert test_entry["file_md5_match"] is False
    assert drift["_pass"] is False
