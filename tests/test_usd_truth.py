"""Tests for scripts/build_usd_truth.py -- the dollar-space truth panel.

The script exists so that a holder of the released USD forecasts can rebuild the
realized values from their own Compustat pull. Its whole risk is producing a
series that *looks* mergeable but is defined differently from the one the
forecasts are of, which is silent by construction: both sides are numeric, both
are $M, and a wrong merge errors nowhere.

So these tests are about definitions and about keys, not plumbing. Each one is
written to fail under a specific mutation of the source -- the fixtures
deliberately carry unpadded gvkeys, NaNs, duplicate format variants and a
missing composite input, because a fixture that is too tidy makes its test
vacuous.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def usd():
    spec = importlib.util.spec_from_file_location(
        "build_usd_truth", ROOT / "scripts" / "build_usd_truth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pull(n_years=2, wcapq_native=999.0, gvkeys=("1690", "2285"), with_filters=True):
    """A fundq-shaped pull. gvkeys are UNPADDED on purpose (see zfill test)."""
    rows = []
    for gvkey, offset in zip(gvkeys, (0, 2)):
        for fyear in range(2020, 2020 + n_years):
            oancf_ytd = 0.0
            for fqtr in (1, 2, 3, 4):
                oancf_ytd += 40.0 + 10 * fqtr
                datadate = (pd.Timestamp(year=fyear, month=3, day=31)
                            + pd.DateOffset(months=3 * (fqtr - 1) + offset))
                row = {
                    "gvkey": gvkey, "fyearq": fyear, "fqtr": fqtr,
                    "datadate": datadate,
                    "revtq": 500.0 + fqtr, "cogsq": 300.0 + fqtr,
                    "actq": 200.0, "lctq": 80.0,
                    "wcapq": wcapq_native,       # native, differently defined
                    "oancfy": oancf_ytd, "capxy": 10.0 * fqtr,
                }
                if with_filters:
                    row |= {"indfmt": "INDL", "datafmt": "STD",
                            "consol": "C", "popsrc": "D"}
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- definitions --

def test_native_wcapq_is_overwritten_not_passed_through(usd):
    """The whole point. Compustat's wcapq merges cleanly and means something else."""
    out = usd.build_usd_truth(_pull(wcapq_native=999.0), targets=["wcapq"])
    assert set(out["target"]) == {"wcapq"}
    np.testing.assert_allclose(out["actual_musd"].to_numpy(), 120.0)  # actq - lctq
    assert 999.0 not in set(out["actual_musd"].to_numpy())


def test_uncomputable_composite_is_dropped_not_passed_through(usd):
    """add_computed_features silently no-ops when an input is missing. Without a
    guard the native column would be emitted as truth -- exactly the trap."""
    pull = _pull(wcapq_native=999.0).drop(columns=["actq", "lctq"])
    out = usd.build_usd_truth(pull)
    assert "wcapq" not in set(out["target"])
    assert 999.0 not in set(out["actual_musd"].to_numpy())


def test_naming_an_uncomputable_composite_is_an_error(usd):
    """--targets wcapq must refuse rather than substitute the native column."""
    pull = _pull(wcapq_native=999.0).drop(columns=["actq", "lctq"])
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["wcapq"])
    assert e.value.code == 2


def test_ytd_source_is_decumulated(usd):
    """oancfq must be the quarterly flow, never the year-to-date level."""
    out = usd.build_usd_truth(_pull(), targets=["oancfq"])
    firm = out[out.firm_id == "001690"].sort_values("quarter")
    assert firm["actual_musd"].iloc[0] == pytest.approx(50.0)
    np.testing.assert_allclose(firm["actual_musd"].to_numpy()[1:4], [60.0, 70.0, 80.0])
    assert "oancfy" not in set(out["target"])


def test_composites_are_computed_after_decumulation(usd):
    """fcfq = oancfq - capxq. Computing before de-cumulation silently uses the
    YTD levels, so this pins the ORDER, not just the presence of the column."""
    out = usd.build_usd_truth(_pull(), targets=["fcfq"])
    firm = out[out.firm_id == "001690"].sort_values("quarter")
    # quarterly oancf 50,60,70,80 minus quarterly capx 10,10,10,10
    np.testing.assert_allclose(firm["actual_musd"].to_numpy()[:4], [40.0, 50.0, 60.0, 70.0])


# ---------------------------------------------------------------------- keys --

def test_gvkey_is_zero_padded(usd):
    """Fixture gvkeys are unpadded, so dropping zfill must break this."""
    out = usd.build_usd_truth(_pull(), targets=["revtq"])
    assert set(out["firm_id"]) == {"001690", "002285"}


def test_float_gvkey_does_not_become_1690_point_0(usd):
    """A CSV pull with any missing gvkey types the column float64; a naive
    astype(str) then yields '1690.0', which is 6 chars so zfill is a no-op and
    the id merges against nothing."""
    pull = _pull()
    pull["gvkey"] = pull["gvkey"].astype(float)
    out = usd.build_usd_truth(pull, targets=["revtq"])
    assert set(out["firm_id"]) == {"001690", "002285"}
    assert not any("." in f for f in out["firm_id"])


def test_non_numeric_gvkey_refuses(usd):
    pull = _pull(gvkeys=("1690", "not-a-gvkey"))
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["revtq"])
    assert e.value.code == 2


def test_quarter_is_calendar_quarter_end(usd):
    out = usd.build_usd_truth(_pull(), targets=["revtq"])
    q = pd.to_datetime(out["quarter"])
    assert q.eq(q.dt.to_period("Q").dt.end_time).all()
    assert q.dt.month.isin({3, 6, 9, 12}).all()


# ------------------------------------------------------------------ hygiene --

def test_nulls_are_dropped(usd):
    """Fixture must contain a NaN or this assertion is vacuous."""
    pull = _pull()
    pull.loc[0, "revtq"] = np.nan
    out = usd.build_usd_truth(pull, targets=["revtq"])
    assert out["actual_musd"].notna().all()
    assert len(out) == len(pull) - 1


def test_restated_firm_quarter_yields_exactly_one_truth(usd):
    """Compustat carries two datadates in one calendar quarter for ~4.6k
    firm-quarters (fiscal-period changes, restatements). Exactly one row must
    survive per key, and it must be the later fiscal period."""
    pull = _pull()
    superseded = pull.iloc[[0]].copy()
    superseded["revtq"] = -999.0
    superseded["fqtr"] = 0                 # sorts before the real fqtr=1 row
    out = usd.build_usd_truth(pd.concat([superseded, pull], ignore_index=True),
                              targets=["revtq"])
    assert not out.duplicated(["firm_id", "quarter", "target"]).any()
    assert -999.0 not in set(out["actual_musd"].to_numpy())
    assert len(out) == len(pull)


def test_long_format_is_typed(usd):
    out = usd.build_usd_truth(_pull())
    assert list(out.columns) == ["firm_id", "quarter", "target", "actual_musd"]
    assert out["actual_musd"].dtype == np.float32
    assert not out.duplicated(["firm_id", "quarter", "target"]).any()


# ------------------------------------------------------------------ filters --

def test_canonical_format_variant_wins_regardless_of_row_order(usd):
    """comp.fundq carries several format variants per firm-quarter. Without the
    filters, de-duplication keeps whichever sorted last -- an artefact of the
    caller's query order, not of the data."""
    good = _pull()
    bad = _pull()
    bad["indfmt"] = "FS"
    bad["revtq"] = 9999.0
    a = usd.build_usd_truth(pd.concat([good, bad], ignore_index=True), targets=["revtq"])
    b = usd.build_usd_truth(pd.concat([bad, good], ignore_index=True), targets=["revtq"])
    assert 9999.0 not in set(a["actual_musd"].to_numpy())
    pd.testing.assert_frame_equal(a, b)


def test_missing_filter_columns_warn_but_proceed(usd, capsys):
    out = usd.build_usd_truth(_pull(with_filters=False), targets=["revtq"])
    assert len(out) > 0
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------- exit codes --

def test_missing_fiscal_fields_refuse_with_code_2(usd):
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(_pull().drop(columns=["fqtr"]))
    assert e.value.code == 2


def test_nothing_derivable_is_code_2_not_an_empty_success(usd):
    """A zero-row parquet and exit 0 reads as success to any wrapper."""
    pull = _pull()[["gvkey", "datadate", "fyearq", "fqtr"]]
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull)
    assert e.value.code == 2


def test_unknown_requested_target_is_an_error(usd):
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(_pull(), targets=["revtq", "not_a_real_item"])
    assert e.value.code == 2


def test_targets_come_from_the_task_config(usd, monkeypatch):
    """Not a hardcoded 'pf_full'.

    Asserting benchmark_targets() == feature_set_items(load_task_config()[...])
    would be vacuous: both sides are pf_full today, so a hardcoded 'pf_full'
    passes it. Point the config at a different feature set and require the
    script to follow.
    """
    from proforma20q.config import feature_set_items, load_task_config

    assert usd.benchmark_targets() == feature_set_items("pf_full")
    assert len(usd.benchmark_targets()) == 78

    real = load_task_config()
    other = "pf_avail"
    assert feature_set_items(other) != feature_set_items("pf_full"), \
        f"{other} no longer differs from pf_full; pick another for this test"
    monkeypatch.setattr(usd, "load_task_config",
                        lambda: {**real, "benchmark": {**real["benchmark"],
                                                       "feature_set": other}})
    assert usd.benchmark_targets() == feature_set_items(other)
