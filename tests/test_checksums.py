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


def test_column_stats_source_preserved_on_rerun(tmp_path, capsys):
    """The provenance note survives a re-run, like download_date/task_version --
    otherwise a routine repopulation silently destroys the only audit trail for
    statistics whose source build no longer exists. But the carry-forward must
    WARN: the note is user-facing report-drift output, so silently keeping it
    across a re-run that just recomputed the stats from a different build would
    print stale provenance as the stated source of statistics it no longer
    describes."""
    proc = tmp_path / "processed"
    proc.mkdir()
    _make_processed(proc)
    out = tmp_path / "checksums.json"

    write_checksums(proc, SUFFIX, out_path=out, download_date="2026-07-02",
                    task_version="r13", column_stats_source="computed from X")
    assert json.loads(out.read_text(encoding="utf-8"))[
        "_column_stats_source"] == "computed from X"
    assert "carrying forward" not in capsys.readouterr().out  # explicit note: quiet
    write_checksums(proc, SUFFIX, out_path=out)  # nothing passed
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["_column_stats_source"] == "computed from X"
    assert "carrying forward _column_stats_source" in capsys.readouterr().out
    # and report_drift surfaces it next to the verdict it decides
    from proforma20q.checksums import report_drift
    rep = report_drift(proc, SUFFIX, published=rec)
    assert rep["_column_stats_source"] == "computed from X"


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


def test_drift_against_a_reference_build(tmp_path):
    """`--reference <dir>` compares against a build you already trust, and the
    statistic separates a same-panel rebuild from a different one. This is the
    property the old per-column-hash metric did not have: it read ~100%
    diverged for both."""
    from proforma20q.checksums import reference_record

    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    published = reference_record(ref, SUFFIX)

    same = tmp_path / "same"
    same.mkdir()
    _make_processed(same)                       # same seeds -> same values
    ok = report_drift(same, SUFFIX, published=published)
    assert ok["_pass"] is True

    other = tmp_path / "other"
    other.mkdir()
    for split, seed in (("train", 91), ("val", 92), ("test", 93)):   # different data
        _tabular_frame(seed).to_parquet(
            other / f"tabular_{split}__{SUFFIX}.parquet", engine=PARQUET_ENGINE, index=False)
    bad = report_drift(other, SUFFIX, published=published)
    assert bad["_pass"] is False
    entry = bad[f"tabular_test__{SUFFIX}.parquet"]
    assert entry["n_cols_over_threshold"] > 0
    assert entry["worst_abs_mean_delta"] > 0.05


def test_drift_refuses_to_compare_a_differently_tagged_build(tmp_path):
    """Reporting every published artifact "missing" and FAILing says nothing
    about drift; it has to say the tags do not match."""
    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    out = tmp_path / "checksums.json"
    write_checksums(ref, SUFFIX, out_path=out, download_date="2026-07-02", task_version="r13")
    published = json.loads(out.read_text(encoding="utf-8"))

    rep = report_drift(ref, "pf_full__some_other_tag", published=published)
    assert rep["_pass"] is None
    assert "nothing to compare" in rep["_note"]


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


def test_drift_does_not_pass_on_a_column_set_mismatch(tmp_path):
    """The symmetric failure to the old metric. `_compare_column_stats`
    intersects the two column sets, so a build sharing NO columns with the
    reference used to score `0/0 columns beyond tolerance` -> PASS, certified on
    similar row counts alone."""
    from proforma20q.checksums import _compare_column_stats, reference_record

    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    published = reference_record(ref, SUFFIX)

    # same shape and row count, entirely different column names
    other = tmp_path / "other"
    other.mkdir()
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        df = _tabular_frame(seed).rename(columns=lambda c: c.replace("niq", "zzz")
                                         .replace("revtq", "yyy"))
        df.to_parquet(other / f"tabular_{split}__{SUFFIX}.parquet",
                      engine=PARQUET_ENGINE, index=False)

    rep = report_drift(other, SUFFIX, published=published)
    entry = rep[f"tabular_test__{SUFFIX}.parquet"]
    assert entry["n_cols"] == 0
    assert entry["row_count_delta"] == 0          # row counts alone look fine
    assert entry["pass"] is False                 # ... and that is not a pass
    assert rep["_pass"] is False
    # the mismatch is visible in the report, not just in the verdict
    assert entry["n_cols_only_here"] > 0 and entry["n_cols_only_published"] > 0
    assert entry["cols_only_published"]

    # a PARTIAL overlap is also not a pass
    partial = tmp_path / "partial"
    partial.mkdir()
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        df = _tabular_frame(seed).drop(columns=["niq_t1"])
        df.to_parquet(partial / f"tabular_{split}__{SUFFIX}.parquet",
                      engine=PARQUET_ENGINE, index=False)
    rep2 = report_drift(partial, SUFFIX, published=published)
    e2 = rep2[f"tabular_test__{SUFFIX}.parquet"]
    assert e2["n_cols"] > 0 and e2["n_cols_over_threshold"] == 0   # values agree
    assert e2["n_cols_only_published"] == 1 and e2["pass"] is False

    # and the helper itself reports the disjoint case honestly
    cmp = _compare_column_stats({"a": {"n_finite": 1, "mean": 0.0, "sd": 1.0,
                                       "coverage": 1.0}},
                                {"b": {"n_finite": 1, "mean": 0.0, "sd": 1.0,
                                       "coverage": 1.0}},
                                {"max_abs_mean_delta": 0.05, "max_abs_sd_delta": 0.05,
                                 "max_abs_coverage_delta": 0.02})
    assert cmp["n_cols"] == 0 and cmp["n_cols_only_here"] == 1


# --------------------------------------------------------------------------- #
# Gated column statistics: an unreachable reference must read NOT VERIFIED.
# --------------------------------------------------------------------------- #
def _published_without_stats(dirpath, tmp_path):
    """A populated published record with the column_stats stripped out.

    This is the shape that actually ships: checksums.json keeps the md5 pins and
    the counts, and the comparable statistics come from the gated repository.
    """
    out = tmp_path / "checksums.json"
    write_checksums(dirpath, SUFFIX, out_path=out,
                    download_date="2026-07-02", task_version="r13")
    rec = json.loads(out.read_text(encoding="utf-8"))
    for entry in rec["artifacts"].values():
        entry.pop("column_stats", None)
    return rec


def _gate_closed(monkeypatch):
    from proforma20q.gated import GatedArtifactError

    def _raise():
        raise GatedArtifactError("Cannot access the gated repository (test)")

    monkeypatch.setattr("proforma20q.checksums.load_canonical_column_stats", _raise)


def test_drift_is_not_verified_when_the_gate_is_unreachable(tmp_path, monkeypatch):
    """The failure mode this guards is silence. Without the canonical
    statistics NOTHING is compared, so the report must say so and must not be a
    pass -- the promise is "verified by published checksums", and an unreachable
    reference means unverified."""
    built = tmp_path / "built"
    built.mkdir()
    _make_processed(built)
    published = _published_without_stats(built, tmp_path)
    _gate_closed(monkeypatch)

    rep = report_drift(built, SUFFIX, published=published)
    assert rep["_pass"] is False, "an unfetchable reference must never read as a pass"
    assert rep["_pass"] is not None, "and must not be indeterminate either"
    assert "_not_verified" in rep
    # The legacy per-column hash metric must not be allowed to stand in as a
    # verdict: it is present in the record but decides nothing.
    assert "_note" not in rep or "NOT VERIFIED" not in rep.get("_note", "")


def test_report_drift_cli_says_not_verified_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """The words matter as much as the exit code: a user who sees anything other
    than NOT VERIFIED will believe the build was checked."""
    from proforma20q.cli import main

    built = tmp_path / "processed"
    built.mkdir()
    _make_processed(built)
    _gate_closed(monkeypatch)

    rc = main(["report-drift", "--processed", str(built)])
    out = capsys.readouterr().out
    assert rc != 0, "an unverified build must not exit 0"
    assert "NOT VERIFIED" in out
    assert "not a pass" in out.lower()


def test_drift_reference_build_needs_no_gate(tmp_path, monkeypatch):
    """`--reference <dir>` computes its statistics locally, so it must keep
    working with the gate closed -- otherwise the offline escape hatch the NOT
    VERIFIED message points at would not exist."""
    from proforma20q.checksums import reference_record

    ref = tmp_path / "ref"
    ref.mkdir()
    _make_processed(ref)
    same = tmp_path / "same"
    same.mkdir()
    _make_processed(same)

    _gate_closed(monkeypatch)
    rep = report_drift(same, SUFFIX, published=reference_record(ref, SUFFIX))
    assert rep["_pass"] is True
    assert "_not_verified" not in rep


def test_attach_column_stats_merges_and_guards_the_suffix(tmp_path):
    """The gated file supplies comparison values only -- never the artifact list,
    the md5s, or statistics computed over a different sample."""
    import pytest

    from proforma20q.checksums import attach_column_stats

    built = tmp_path / "built"
    built.mkdir()
    _make_processed(built)
    full = json.loads(
        (write_checksums(built, SUFFIX, out_path=tmp_path / "c.json")).read_text(encoding="utf-8"))
    stripped = json.loads(json.dumps(full))
    canonical = {"suffix": SUFFIX, "artifacts": {}}
    for name, entry in stripped["artifacts"].items():
        cs = entry.pop("column_stats", None)
        if cs is not None:
            canonical["artifacts"][name] = {"n_rows": entry["n_rows"], "column_stats": cs}

    merged = attach_column_stats(stripped, canonical)
    for name, entry in full["artifacts"].items():
        if "column_stats" in entry:
            assert merged["artifacts"][name]["column_stats"] == entry["column_stats"]
        assert merged["artifacts"][name]["md5"] == entry["md5"]

    # statistics computed over a different sample are not a reference for this one
    other = json.loads(json.dumps(canonical))
    for centry in other["artifacts"].values():
        centry["n_rows"] = centry["n_rows"] + 1
    dropped = attach_column_stats(json.loads(json.dumps(stripped)), other)
    assert all("column_stats" not in e for e in dropped["artifacts"].values())

    with pytest.raises(ValueError, match="suffix"):
        attach_column_stats(stripped, {"suffix": "pf_full__other", "artifacts": {}})
