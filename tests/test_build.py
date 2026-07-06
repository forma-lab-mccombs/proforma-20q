"""End-to-end build on synthetic data: proves the pipeline is wired correctly
even though the real build needs WRDS."""
import numpy as np
import pandas as pd
import pytest

from proforma20q.build import add_computed_features, build
from proforma20q.baselines import run_baseline
from proforma20q.baselines.common import discover_targets, load_tabular
from proforma20q.evaluate import evaluate_forecasts
from fixtures import synthetic_raw


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("build")
    raw = synthetic_raw()
    raw_path = tmp / "compustat_with_permno.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)
    out = build(raw_path, tmp / "processed", dataset_tag="test", verbose=False)
    return out


def test_all_artifacts_written(built):
    for key in ("reg_stats", "tabular_train", "tabular_val", "tabular_test",
                "tuple_train", "tuple_val", "tuple_test"):
        assert key in built and built[key].exists()


def test_tabular_column_layout(built):
    test = load_tabular(built["tabular_test"])
    assert "scale_level_0" in test.columns
    for item in ("niq", "revtq", "gpq", "fcfq"):
        assert f"{item}_level_0" in test.columns
        assert f"{item}_level_3" in test.columns
        assert f"{item}_yoy_0" in test.columns
        assert f"{item}_yoy_7" in test.columns
        assert f"{item}_t1" in test.columns
        assert f"{item}_t20" in test.columns
    ff = [c for c in test.columns if c.startswith("indff48_")]
    assert len(ff) == 48  # ids 0..47, unknown_id=48 dropped as reference


def test_regularized_values_clamped(built):
    test = load_tabular(built["tabular_test"])
    zcols = [c for c in test.columns
             if ("_level_" in c or (c.rsplit("_t", 1)[-1].isdigit() and "_t" in c))
             and "_yoy_" not in c]
    vals = test[zcols].to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.min() >= -6.0 - 1e-6
    assert finite.max() <= 6.0 + 1e-6


def test_splits_are_calendar_disjoint(built):
    tr = load_tabular(built["tabular_train"])
    te = load_tabular(built["tabular_test"])
    tr_years = pd.to_datetime(tr["origin"]).dt.year
    te_years = pd.to_datetime(te["origin"]).dt.year
    assert tr_years.max() <= 2001
    assert te_years.min() >= 2010


def test_naive_is_change_space_anchor(built):
    train = load_tabular(built["tabular_train"])
    val = load_tabular(built["tabular_val"])
    test = load_tabular(built["tabular_test"])
    targets = discover_targets(test)
    naive = run_baseline("naive", train, test, val_df=val, targets=targets)
    res = evaluate_forecasts({"naive": naive}, test, verbose=False)
    r2 = res.leaderboard()[lambda d: d.model == "naive"]["r2"].iloc[0]
    # RW predicts zero change -> its change-space R2 is ~0 by construction.
    assert abs(r2) < 0.05


def test_fade_beats_naive_under_mean_reversion(built):
    train = load_tabular(built["tabular_train"])
    val = load_tabular(built["tabular_val"])
    test = load_tabular(built["tabular_test"])
    targets = discover_targets(test)
    naive = run_baseline("naive", train, test, val_df=val, targets=targets)
    fade = run_baseline("fade", train, test, val_df=val, targets=targets)
    res = evaluate_forecasts({"naive": naive, "fade": fade}, test, verbose=False)
    lb = res.leaderboard()
    r2_naive = lb[lb.model == "naive"]["r2"].iloc[0]
    r2_fade = lb[lb.model == "fade"]["r2"].iloc[0]
    assert r2_fade >= r2_naive


def test_tuple_schema(built):
    tup = pd.read_parquet(built["tuple_test"], engine="fastparquet")
    assert set(tup.columns) == {"firm_id", "account_id", "quarter", "value", "industry_id"}
    assert tup["firm_id"].dtype.kind == "i"
    assert tup["quarter"].dtype.kind == "i"


def test_computed_features_overwrite_native_columns():
    """wcapq/gpq/dvcq ship natively in comp.fundq but the benchmark DEFINES them
    via formula; the builder must overwrite the native column (matching the
    research repo) rather than keep the as-reported value."""
    df = pd.DataFrame({
        "actq": [10.0, 20.0], "lctq": [4.0, 5.0],
        "wcapq": [999.0, 999.0],  # bogus native value that must be overwritten
        "revtq": [100.0, 200.0], "cogsq": [60.0, 150.0],
    })
    out = add_computed_features(df, ["wcapq", "gpq"])
    np.testing.assert_allclose(out["wcapq"].to_numpy(), [6.0, 15.0])  # actq - lctq, not 999
    np.testing.assert_allclose(out["gpq"].to_numpy(), [40.0, 50.0])   # revtq - cogsq


def test_regularized_columns_are_float32(built):
    test = load_tabular(built["tabular_test"])
    for c in ("scale_level_0", "niq_level_0", "niq_yoy_0", "niq_t1"):
        assert test[c].dtype == np.float32, f"{c} should be float32 (matches research repo)"


def test_frozen_reg_stats_are_consumed(tmp_path):
    """Passing reg_stats pins normalization: perturbing the frozen stats must move
    the regularized output, proving they are used instead of re-estimated."""
    raw = synthetic_raw()
    raw_path = tmp_path / "compustat_with_permno.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)

    base = build(raw_path, tmp_path / "p_est", dataset_tag="est", which=("tabular",), verbose=False)
    a = load_tabular(base["tabular_test"]).sort_values(["firm", "origin"]).reset_index(drop=True)

    frozen = pd.read_parquet(base["reg_stats"], engine="fastparquet")
    frozen["mu"] = frozen["mu"] + 1.0  # deliberate shift
    frozen_path = tmp_path / "frozen_reg_stats.parquet"
    frozen.to_parquet(frozen_path, engine="fastparquet", index=False)

    frz = build(raw_path, tmp_path / "p_frz", dataset_tag="frz", which=("tabular",),
                reg_stats=frozen_path, verbose=False)
    b = load_tabular(frz["tabular_test"]).sort_values(["firm", "origin"]).reset_index(drop=True)

    # The written reg_stats artifact is the frozen one (round-tripped), not a re-estimate.
    written = pd.read_parquet(frz["reg_stats"], engine="fastparquet")
    np.testing.assert_allclose(written["mu"].to_numpy(), frozen["mu"].to_numpy())

    # A shifted mu changes the normalized level_0 for at least some cells.
    va, vb = a["niq_level_0"].to_numpy(), b["niq_level_0"].to_numpy()
    both = np.isfinite(va) & np.isfinite(vb)
    assert both.any() and not np.allclose(va[both], vb[both])
