"""Tests for scripts/build_usd_truth.py -- the dollar-space truth panel.

The script exists so that a holder of the released USD forecasts can rebuild the
realized values from their own Compustat pull. Its whole risk is producing a
series that *looks* mergeable but is defined differently from the one the
forecasts are of, which is silent by construction: both sides are numeric, both
are $M, and a wrong merge errors nowhere.

So these tests are about definitions, not plumbing: that the native ``wcapq``
column is overwritten rather than passed through, that YTD sources are
de-cumulated rather than emitted as levels, and that the keys line up with the
release. The formulas themselves are the build's and are tested there.
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


def _pull(n_years=2, wcapq_native=999.0):
    """A tiny fundq-shaped pull: one calendar-year firm, one June-FYE firm."""
    rows = []
    for gvkey, offset in (("001690", 0), ("002285", 2)):
        for fyear in range(2020, 2020 + n_years):
            oancf_ytd = 0.0
            for fqtr in (1, 2, 3, 4):
                oancf_ytd += 40.0 + 10 * fqtr
                datadate = (pd.Timestamp(year=fyear, month=3, day=31)
                            + pd.DateOffset(months=3 * (fqtr - 1) + offset))
                rows.append({
                    "gvkey": gvkey, "fyearq": fyear, "fqtr": fqtr,
                    "datadate": datadate,
                    "revtq": 500.0 + fqtr, "cogsq": 300.0 + fqtr,
                    "actq": 200.0, "lctq": 80.0,
                    "wcapq": wcapq_native,       # native column, differently defined
                    "oancfy": oancf_ytd,
                })
    return pd.DataFrame(rows)


def test_native_wcapq_is_overwritten_not_passed_through(usd):
    """The whole point. Compustat's wcapq merges cleanly and means something else."""
    out = usd.build_usd_truth(_pull(wcapq_native=999.0), targets=["wcapq"])
    assert len(out) > 0
    assert set(out["target"]) == {"wcapq"}
    np.testing.assert_allclose(out["actual_musd"].to_numpy(), 120.0)  # actq - lctq
    assert 999.0 not in set(out["actual_musd"].to_numpy())


def test_ytd_source_is_decumulated(usd):
    """oancfq must be the quarterly flow, never the year-to-date level."""
    out = usd.build_usd_truth(_pull(), targets=["oancfq"])
    firm = out[out.firm_id == "001690"].sort_values("quarter")
    # Q1 keeps the YTD value; later quarters are first differences within the year
    assert firm["actual_musd"].iloc[0] == pytest.approx(50.0)
    np.testing.assert_allclose(firm["actual_musd"].to_numpy()[1:4],
                               [60.0, 70.0, 80.0])
    assert "oancfy" not in set(out["target"])


def test_keys_match_the_release(usd):
    """gvkey zero-padded to 6, datadate snapped to calendar quarter end."""
    out = usd.build_usd_truth(_pull(), targets=["revtq"])
    assert set(out["firm_id"]) == {"001690", "002285"}
    assert out["firm_id"].map(len).eq(6).all()
    q = pd.to_datetime(out["quarter"])
    assert q.eq(q.dt.to_period("Q").dt.end_time).all()
    # the June-FYE firm's fiscal Q1 lands in a calendar quarter, not on datadate
    assert q.dt.month.isin({3, 6, 9, 12}).all()


def test_long_format_is_unique_and_typed(usd):
    out = usd.build_usd_truth(_pull())
    assert list(out.columns) == ["firm_id", "quarter", "target", "actual_musd"]
    assert not out.duplicated(["firm_id", "quarter", "target"]).any()
    assert out["actual_musd"].dtype == np.float32
    assert out["actual_musd"].notna().all()


def test_computed_targets_are_all_covered(usd):
    """A pull carrying every input must yield every computed pf_full item -- the
    regression guarding against a composite being quietly left out."""
    from proforma20q.build import _computed_definitions
    from proforma20q.config import pf_full_targets

    computed = {t for t in pf_full_targets() if t in _computed_definitions()}
    assert "wcapq" in computed, "pf_full lost wcapq; this test is now wrong"

    inputs = sorted({c for t in computed for c in _computed_definitions()[t][0]})
    pull = _pull()
    for c in inputs:
        if c not in pull.columns:
            pull[c] = 10.0

    out = usd.build_usd_truth(pull, targets=sorted(computed))
    assert set(out["target"]) == computed


def test_missing_fiscal_fields_refuse_rather_than_guess(usd):
    pull = _pull().drop(columns=["fqtr"])
    with pytest.raises(SystemExit, match="fqtr"):
        usd.build_usd_truth(pull)


def test_unknown_requested_target_is_an_error(usd):
    with pytest.raises(SystemExit, match="not derivable"):
        usd.build_usd_truth(_pull(), targets=["revtq", "not_a_real_item"])
