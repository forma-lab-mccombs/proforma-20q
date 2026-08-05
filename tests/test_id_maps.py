"""The pinned canonical id maps and the id-map drift check.

The tuple view's account/industry integer ids are embedding indices; the
pinned reference CSVs in ``reference/`` exist so a rebuild can prove its
orderings match the canonical build's (a silent permutation is the failure
mode). All synthetic -- no WRDS data.
"""
from __future__ import annotations

import json

import pandas as pd

from proforma20q.build import (
    CANONICAL_N_FIRMS,
    account_id_map_table,
    canonical_account_id_map,
    industry_id_map_table,
    verify_id_maps,
    write_id_maps,
)
from proforma20q.config import canonical_id_map_path, load_ff48_ranges

SUFFIX = "pf_full__r13_node_optionD_indfe_val8"


def _canonical_firm_map() -> dict[str, int]:
    """A synthetic firm map with the canonical count and ordering rule."""
    return {f"{i:06d}": i for i in range(CANONICAL_N_FIRMS)}


# --------------------------------------------------------------------------- #
# The packaged reference CSVs
# --------------------------------------------------------------------------- #
def test_packaged_account_reference_is_the_canonical_79_entry_map():
    p = canonical_id_map_path("account_id_map")
    assert p.exists(), "canonical account_id_map CSV must ship in reference/"
    ref = pd.read_csv(p)
    assert len(ref) == 79, "78 pf_full items plus the scale deflator"
    assert ref["account_id"].tolist() == list(range(79))
    names = ref["account_name"].tolist()
    assert names == sorted(names), "ids are assigned by sorted()"
    # scale sits at id 60, shifting the 18 items after it by +1 -- the exact
    # off-by-one a bare-78-item enumeration would get wrong
    assert names[60] == "scale"
    assert names[59] == "revtq" and names[61] == "seqq"
    # and it is exactly what the build's own map constructor produces
    assert names == list(canonical_account_id_map())


def test_packaged_industry_reference_matches_the_pinned_ff48_table():
    p = canonical_id_map_path("industry_id_map")
    assert p.exists(), "canonical industry_id_map CSV must ship in reference/"
    ref = pd.read_csv(p)
    cfg = load_ff48_ranges()
    by_id = {ind["id"]: ind["name"] for ind in cfg["industries"]}
    assert len(ref) == len(by_id) + 1  # the 48 named + the unknown level
    unknown_id = int(cfg.get("unknown_id", 48))
    for _, row in ref.iterrows():
        iid = int(row["industry_id"])
        if iid == unknown_id:
            assert bool(row["is_reference_level"]) is True
        else:
            assert row["industry_name"] == by_id[iid]
    # regenerating from config reproduces the pinned file exactly
    pd.testing.assert_frame_equal(
        ref, industry_id_map_table(), check_dtype=False)


def test_account_table_roundtrip():
    amap = canonical_account_id_map()
    table = account_id_map_table(amap)
    assert table["account_id"].tolist() == sorted(amap.values())
    assert dict(zip(table["account_name"], table["account_id"])) == amap


# --------------------------------------------------------------------------- #
# verify_id_maps
# --------------------------------------------------------------------------- #
def test_verify_ok_on_a_canonical_ordering_build(tmp_path):
    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), canonical_account_id_map())
    rep = verify_id_maps(tmp_path, SUFFIX)
    assert rep is not None and rep["_ok"] is True
    assert rep["account_id_map"]["status"] == "ok"
    assert rep["industry_id_map"]["status"] == "ok"
    assert rep["firm_id_map"]["status"] == "ok"
    assert rep["firm_id_map"]["ordering_rule_ok"] is True
    assert rep["firm_id_map"]["delta_frac"] == 0.0


def test_verify_returns_none_when_no_maps_written(tmp_path):
    """A tabular-only build has no id maps; that is not a failure."""
    assert verify_id_maps(tmp_path, SUFFIX) is None


def test_verify_fails_on_a_permuted_account_map(tmp_path):
    amap = canonical_account_id_map()
    # drop `scale` and re-enumerate: the exact 78-item map an unaware
    # implementation would build -- every id after 59 shifts down by one
    permuted = {a: i for i, a in enumerate(sorted(a for a in amap if a != "scale"))}
    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), permuted)
    rep = verify_id_maps(tmp_path, SUFFIX)
    assert rep["_ok"] is False
    assert rep["account_id_map"]["status"] == "mismatch"
    assert rep["account_id_map"]["n_built"] == 78
    assert rep["account_id_map"]["n_reference"] == 79
    assert rep["account_id_map"]["first_diffs"]
    # the other two maps are still fine
    assert rep["industry_id_map"]["status"] == "ok"


def test_verify_fails_on_a_firm_ordering_violation(tmp_path):
    firms = _canonical_firm_map()
    # swap two ids: gvkeys no longer sorted by id -- a positional permutation
    firms["000000"], firms["000001"] = 1, 0
    write_id_maps(tmp_path, SUFFIX, firms, canonical_account_id_map())
    rep = verify_id_maps(tmp_path, SUFFIX)
    assert rep["_ok"] is False
    assert rep["firm_id_map"]["status"] == "mismatch"
    assert rep["firm_id_map"]["ordering_rule_ok"] is False


def test_verify_fails_when_the_firm_universe_drifts_too_far(tmp_path):
    firms = {f"{i:06d}": i for i in range(int(CANONICAL_N_FIRMS * 0.9))}  # -10%
    write_id_maps(tmp_path, SUFFIX, firms, canonical_account_id_map())
    rep = verify_id_maps(tmp_path, SUFFIX)
    assert rep["_ok"] is False
    assert rep["firm_id_map"]["status"] == "drift_exceeded"
    # vintage-scale drift is fine
    firms = {f"{i:06d}": i for i in range(CANONICAL_N_FIRMS + 6)}   # the measured +6
    write_id_maps(tmp_path, SUFFIX, firms, canonical_account_id_map())
    assert verify_id_maps(tmp_path, SUFFIX)["_ok"] is True


def test_verify_against_a_reference_dir(tmp_path):
    """`--reference <dir>` scopes to the FIRM map only: its gvkey universe is
    vintage-local, so a trusted build's count is the better comparand there."""
    ref = tmp_path / "ref"
    ref.mkdir()
    write_id_maps(ref, SUFFIX, {"000001": 0, "000002": 1}, canonical_account_id_map())
    same = tmp_path / "same"
    same.mkdir()
    write_id_maps(same, SUFFIX, {"000001": 0, "000003": 1}, canonical_account_id_map())
    rep = verify_id_maps(same, SUFFIX, reference_dir=ref)
    assert rep["_ok"] is True                       # same count, rule holds
    assert rep["firm_id_map"]["n_reference"] == 2


def test_reference_dir_cannot_weaken_the_account_check(tmp_path):
    """Account/industry are ALWAYS compared against the pinned canonical CSVs,
    even under `--reference`. The pins are data-independent -- there is no
    vintage under which they don't apply -- and comparing against a reference
    build's own maps instead would let a permuted reference build agree with an
    identically permuted rebuild: the exact silent failure the pins exist to
    catch."""
    amap = canonical_account_id_map()
    permuted = {a: i for i, a in enumerate(sorted(a for a in amap if a != "scale"))}
    ref = tmp_path / "ref"
    ref.mkdir()
    write_id_maps(ref, SUFFIX, {"000001": 0}, permuted)     # permuted reference...
    rebuild = tmp_path / "rebuild"
    rebuild.mkdir()
    write_id_maps(rebuild, SUFFIX, {"000001": 0}, permuted)  # ...agrees with rebuild
    rep = verify_id_maps(rebuild, SUFFIX, reference_dir=ref)
    assert rep["account_id_map"]["status"] == "mismatch"
    assert rep["_ok"] is False


def test_no_reference_is_reported_but_not_a_failure(tmp_path, monkeypatch):
    """A feature set with no pinned maps: nothing to compare is not a FAIL."""
    import proforma20q.config as config
    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), canonical_account_id_map())
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path / "empty")
    rep = verify_id_maps(tmp_path, SUFFIX)
    assert rep["account_id_map"]["status"] == "no_reference"
    assert rep["industry_id_map"]["status"] == "no_reference"
    assert rep["_ok"] is True


# --------------------------------------------------------------------------- #
# The build's end-of-run id-map lines
# --------------------------------------------------------------------------- #
def test_every_id_map_failure_mode_produces_build_output(tmp_path):
    """A build must never end silently on a state report-drift would FAIL --
    every not-ok verify_id_maps report has to yield a WARNING line."""
    from proforma20q.build import id_map_log_lines

    # all ok -> a single confirmation line naming the firm count
    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), canonical_account_id_map())
    lines = id_map_log_lines(verify_id_maps(tmp_path, SUFFIX))
    assert len(lines) == 1 and "match the pinned canonical reference" in lines[0]
    assert f"{CANONICAL_N_FIRMS:,}" in lines[0]

    # firm-map drift beyond tolerance (account/industry still fine) -> WARNING
    shrunk = {f"{i:06d}": i for i in range(int(CANONICAL_N_FIRMS * 0.9))}
    write_id_maps(tmp_path, SUFFIX, shrunk, canonical_account_id_map())
    lines = id_map_log_lines(verify_id_maps(tmp_path, SUFFIX))
    assert any("WARNING" in ln for ln in lines)
    assert any("firm_id_map: drift_exceeded" in ln for ln in lines)

    # permuted account map -> WARNING naming the map
    amap = canonical_account_id_map()
    permuted = {a: i for i, a in enumerate(sorted(a for a in amap if a != "scale"))}
    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), permuted)
    lines = id_map_log_lines(verify_id_maps(tmp_path, SUFFIX))
    assert any("WARNING" in ln for ln in lines)
    assert any("account_id_map: mismatch" in ln for ln in lines)


def test_no_reference_build_output_says_so(tmp_path, monkeypatch):
    from proforma20q.build import id_map_log_lines
    import proforma20q.config as config

    write_id_maps(tmp_path, SUFFIX, _canonical_firm_map(), canonical_account_id_map())
    monkeypatch.setattr(config, "REFERENCE_DIR", tmp_path / "empty")
    lines = id_map_log_lines(verify_id_maps(tmp_path, SUFFIX))
    assert len(lines) == 1
    assert "no pinned canonical reference" in lines[0]
    assert "ordering rule ok" in lines[0]


# --------------------------------------------------------------------------- #
# report_drift integration
# --------------------------------------------------------------------------- #
def _make_tabular(dirpath, seed_shift=0):
    import numpy as np
    from proforma20q.schema import PARQUET_ENGINE
    for split, seed in (("train", 1), ("val", 2), ("test", 3)):
        rng = np.random.default_rng(seed + seed_shift)
        n = 40
        pd.DataFrame({
            "firm_id": [f"{i:05d}" for i in range(n)],
            "niq_level_0": rng.standard_normal(n),
            "niq_t1": rng.standard_normal(n),
        }).to_parquet(dirpath / f"tabular_{split}__{SUFFIX}.parquet",
                      engine=PARQUET_ENGINE, index=False)


def test_report_drift_fails_on_permuted_id_maps_even_when_stats_pass(tmp_path):
    from proforma20q.checksums import report_drift, write_checksums

    proc = tmp_path / "processed"
    proc.mkdir()
    _make_tabular(proc)
    out = tmp_path / "checksums.json"
    write_checksums(proc, SUFFIX, out_path=out,
                    download_date="2026-07-02", task_version="r13")
    published = json.loads(out.read_text(encoding="utf-8"))

    # canonical-ordering maps: verdict is PASS
    write_id_maps(proc, SUFFIX, _canonical_firm_map(), canonical_account_id_map())
    rep = report_drift(proc, SUFFIX, published=published)
    assert rep["_id_maps"]["_ok"] is True
    assert rep["_pass"] is True

    # permute the account map: the stats still match bit-for-bit, but the
    # verdict must flip -- this is the failure the reference maps exist to catch
    amap = canonical_account_id_map()
    permuted = {a: i for i, a in enumerate(sorted(a for a in amap if a != "scale"))}
    write_id_maps(proc, SUFFIX, _canonical_firm_map(), permuted)
    rep = report_drift(proc, SUFFIX, published=published)
    assert rep["_id_maps"]["_ok"] is False
    assert rep["_pass"] is False
