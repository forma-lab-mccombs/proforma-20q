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
import subprocess
import sys
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


def test_naming_an_uncomputable_composite_is_an_error(usd, capsys):
    """--targets wcapq must refuse rather than substitute the native column.

    Asserting only the exit code is not enough: several exit-2 guards overlap,
    so deleting this one still yields 2 via 'no targets derivable'. Pin the
    message, and pair it with a request that would otherwise succeed -- under
    that mutation `wcapq,revtq` silently emits revtq alone and exits 0.
    """
    pull = _pull(wcapq_native=999.0).drop(columns=["actq", "lctq"])
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["wcapq", "revtq"])
    assert e.value.code == 2
    assert "NOT the same series" in capsys.readouterr().err


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


def test_nothing_derivable_is_code_2_not_an_empty_success(usd, capsys):
    """A zero-row parquet and exit 0 reads as success to any wrapper."""
    pull = _pull()[["gvkey", "datadate", "fyearq", "fqtr"]]
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull)
    assert e.value.code == 2
    assert "no targets derivable" in capsys.readouterr().err


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


# ------------------------------------------------------ adversarial round 2 --

def test_mixed_gvkey_padding_is_one_firm_not_a_collision(usd):
    """'1690' and '001690' are the same firm. Normalizing AFTER de-duplication
    would let both through the (firm, quarter) dedup and collide at the end --
    surfacing as an AssertionError blaming upstream, on ordinary input (a CSV
    chunk concatenated with a parquet one)."""
    a = _pull(gvkeys=("1690", "2285"))
    b = _pull(gvkeys=("001690", "002285"))
    b["revtq"] = b["revtq"] + 1.0                 # the later row must win
    out = usd.build_usd_truth(pd.concat([a, b], ignore_index=True), targets=["revtq"])
    assert set(out["firm_id"]) == {"001690", "002285"}
    assert not out.duplicated(["firm_id", "quarter", "target"]).any()
    assert len(out) == len(a)


def test_repeated_target_does_not_duplicate_rows(usd):
    out = usd.build_usd_truth(_pull(), targets=["revtq", "revtq"])
    assert not out.duplicated(["firm_id", "quarter", "target"]).any()
    assert set(out["target"]) == {"revtq"}


def test_native_ytd_flow_is_not_passed_through(usd, capsys):
    """convert_ytd_to_quarterly skips a base whose {base}y is absent -- just as
    silently as add_computed_features. A pull carrying a native oancfq and no
    oancfy must not have it emitted as the de-cumulated series."""
    pull = _pull().drop(columns=["oancfy"])
    pull["oancfq"] = 42.0
    out = usd.build_usd_truth(pull)
    assert "oancfq" not in set(out["target"])
    assert 42.0 not in set(out["actual_musd"].to_numpy())
    assert "year-to-date source" in capsys.readouterr().out


def test_naming_a_native_ytd_flow_is_an_error(usd, capsys):
    pull = _pull().drop(columns=["oancfy"])
    pull["oancfq"] = 42.0
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["oancfq"])
    assert e.value.code == 2
    assert "year-to-date source" in capsys.readouterr().err


def test_non_benchmark_target_is_refused(usd, capsys):
    """_computed_definitions() is wider than the benchmark (dvcq, neiq), and
    dvcq ships natively in fundq. Without a universe check the guard vouches for
    a column add_computed_features never touched."""
    pull = _pull()
    pull["dvq"], pull["dvpq"], pull["dvcq"] = 50.0, 5.0, 88888.0
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["dvcq"])
    assert e.value.code == 2
    assert "not benchmark targets" in capsys.readouterr().err


def test_non_benchmark_columns_never_leak_into_the_default_run(usd):
    pull = _pull()
    pull["dvq"], pull["dvpq"], pull["dvcq"] = 50.0, 5.0, 88888.0
    out = usd.build_usd_truth(pull)
    assert "dvcq" not in set(out["target"])
    assert "fyearq" not in set(out["target"])
    assert set(out["target"]) <= set(usd.benchmark_targets())


def test_requested_but_all_nan_target_is_an_error(usd, capsys):
    """Derivable is not the same as emitted; a named target must not vanish."""
    pull = _pull()
    pull["actq"] = np.nan                          # wcapq computes to all-NaN
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["wcapq", "revtq"])
    assert e.value.code == 2
    assert "entirely missing" in capsys.readouterr().err


def test_null_gvkey_rows_are_dropped_not_fatal(usd, capsys):
    pull = _pull()
    pull.loc[0, "gvkey"] = None
    out = usd.build_usd_truth(pull, targets=["revtq"])
    assert len(out) == len(pull) - 1
    assert "no gvkey" in capsys.readouterr().out


def test_filter_values_tolerate_whitespace(usd):
    pull = _pull()
    pull["indfmt"] = " INDL "
    out = usd.build_usd_truth(pull, targets=["revtq"])
    assert len(out) == len(pull)


def test_diagnostics_precede_the_error_in_a_redirected_log(tmp_path):
    """Buffering is only observable out of process. stdout through a pipe is
    block-buffered while stderr is not, so unflushed diagnostics land AFTER the
    error that they explain -- AGENTS.md section 4.

    Two mechanisms defend this and no single mutation separates them: say()
    flushes each line, and Precondition flushes stdout before writing stderr.
    They are kept deliberately, for independent reasons -- say()'s flush buys
    live progress during a long run (which this test does not cover), while
    Precondition's holds the ordering even for a print() that some future edit
    forgets to route through say(). Removing EITHER leaves this test passing;
    removing both fails it. Do not delete one as dead code."""
    pull = _pull()[["gvkey", "datadate", "fyearq", "fqtr"]]
    src = tmp_path / "pull.parquet"
    pull.to_parquet(src, index=False)
    log = tmp_path / "run.log"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_usd_truth.py"),
             "--compustat", str(src), "--out", str(tmp_path / "out.parquet")],
            stdout=fh, stderr=fh, cwd=str(ROOT)).returncode
    assert rc == 2
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    errors = [i for i, ln in enumerate(lines) if ln.startswith("error:")]
    assert errors == [len(lines) - 1], (
        "the error must be the last line, after the SKIP lines explaining it:\n"
        + "\n".join(lines))


# ------------------------------------------------------ adversarial round 3 --

def _pull_native_flows(with_ytd=()):
    """A pull carrying NATIVE oancfq/capxq -- the columns de-cumulation would
    otherwise produce -- and only the YTD sources named in `with_ytd`."""
    pull = _pull().drop(columns=["oancfy", "capxy"])
    pull["oancfq"] = [100.0, 200.0, 300.0, 400.0] * (len(pull) // 4)
    pull["capxq"] = [10.0, 20.0, 30.0, 40.0] * (len(pull) // 4)
    for base in with_ytd:
        pull[f"{base}y"] = 25.0
    return pull


def test_fcfq_does_not_launder_native_flows_past_the_ytd_guard(usd):
    """Derivability is a property of the TRANSITIVE input closure. fcfq's two
    inputs are both YTD-derived, so a one-level requires-check passes while the
    formula consumes columns the de-cumulation never produced -- and the script
    would refuse oancfq while happily emitting oancfq - capxq."""
    out = usd.build_usd_truth(_pull_native_flows())
    assert "fcfq" not in set(out["target"])
    assert 90.0 not in set(out["actual_musd"].to_numpy())


def test_naming_fcfq_with_native_flows_is_an_error(usd, capsys):
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(_pull_native_flows(), targets=["fcfq"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "fcfq" in err and "year-to-date source" in err


def test_fcfq_refused_when_only_one_input_is_derivable(usd):
    """The half-case is the quietest: oancfy present, capxy absent, native
    capxq standing in -- the difference is wrong but entirely plausible."""
    out = usd.build_usd_truth(_pull_native_flows(with_ytd=("oancf",)))
    assert "fcfq" not in set(out["target"])


def test_fcfq_is_emitted_when_both_sources_are_genuinely_present(usd):
    """The guard must not be so strict that it refuses a correct pull."""
    out = usd.build_usd_truth(_pull(), targets=["fcfq"])
    assert set(out["target"]) == {"fcfq"}
    firm = out[out.firm_id == "001690"].sort_values("quarter")
    np.testing.assert_allclose(firm["actual_musd"].to_numpy()[:4], [40.0, 50.0, 60.0, 70.0])


def test_excluded_rows_cannot_abort_the_run_on_their_gvkey(usd):
    """Normalization must follow the format filter: a junk gvkey on a row the
    benchmark's own universe discards should not fail an otherwise-good pull.
    (It must still precede de-duplication -- see the mixed-padding test.)"""
    good = _pull()
    junk = _pull().iloc[[0]].copy()
    junk["indfmt"] = "FS"
    junk["gvkey"] = "TOTAL"
    out = usd.build_usd_truth(pd.concat([good, junk], ignore_index=True),
                              targets=["revtq"])
    assert len(out) == len(good)
    assert set(out["firm_id"]) == {"001690", "002285"}


def test_junk_gvkey_on_an_included_row_still_refuses(usd):
    pull = _pull()
    pull.loc[0, "gvkey"] = "TOTAL"
    with pytest.raises(SystemExit) as e:
        usd.build_usd_truth(pull, targets=["revtq"])
    assert e.value.code == 2
