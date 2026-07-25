import numpy as np
import pandas as pd

from proforma20q.config import feature_set_items, load_task_config, pf_full_targets
from proforma20q.baselines import BASELINES, iter_baseline_blocks, run_baseline, run_baselines
from proforma20q.baselines.common import (discover_targets, load_tabular, target_columns,
                                          wide_predictions_to_long)
from proforma20q.build import build
from proforma20q.evaluate import evaluate_forecasts
from proforma20q.schema import read_forecast
from fixtures import synthetic_raw


def test_pf_full_has_78_targets():
    assert len(pf_full_targets()) == 78


def test_task_config_matches_paper():
    task = load_task_config()
    assert task["splits"]["train_end_year"] == 2001
    assert task["splits"]["val_end_year"] == 2009
    assert task["horizons"] == list(range(1, 21))
    assert task["feature_engineering"]["max_abs_zscore"] == 6.0
    assert task["feature_engineering"]["recent_levels"] == 4
    assert task["feature_engineering"]["yoy_changes"] == 8


def test_baselines_cli_missing_build_exits_2(tmp_path, capsys):
    """`proforma20q baselines` with no build prints a clean one-liner and exits 2
    (like build/evaluate), not a raw traceback."""
    from proforma20q.cli import main

    empty = tmp_path / "processed"
    empty.mkdir()
    rc = main(["baselines", "--processed", str(empty), "--out", str(tmp_path / "fc")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing tabular split" in err
    assert "proforma20q build" in err


def test_wide_to_long_matches_a_melt():
    """`wide_predictions_to_long` no longer melts (549M rows raises
    ArrayMemoryError at benchmark scale); the block-concat replacement must
    reproduce the melt row-for-row, including row order and dtypes."""
    meta = pd.DataFrame({"firm": ["a", "b", "c"],
                         "origin": pd.to_datetime(["2010-03-31"] * 3)})
    wide = pd.DataFrame({"niq_t1": [1.0, 2.0, 3.0], "niq_t2": [4.0, 5.0, 6.0],
                         "atq_t1": [7.0, 8.0, 9.0]})
    tcols = ["atq_t1", "niq_t1", "niq_t2"]

    frame = wide[tcols].copy()
    frame["firm"] = meta["firm"].to_numpy()
    frame["origin"] = meta["origin"].to_numpy()
    long = frame.melt(id_vars=["firm", "origin"], value_vars=tcols,
                      var_name="_tcol", value_name="prediction")
    split = long["_tcol"].str.rsplit("_t", n=1, expand=True)
    long["target"] = split[0]
    long["horizon"] = split[1].astype(int)
    expected = long[["firm", "target", "origin", "horizon", "prediction"]]

    pd.testing.assert_frame_equal(wide_predictions_to_long(meta, wide, tcols), expected)


def _built_splits(tmp_path):
    raw = synthetic_raw(n_firms=20)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)
    out = build(raw_path, tmp_path / "processed", dataset_tag="t",
                which=("tabular",), verbose=False)
    return out, (load_tabular(out["tabular_train"]), load_tabular(out["tabular_val"]),
                 load_tabular(out["tabular_test"]))


def test_streamed_baseline_blocks_reassemble_the_whole_forecast(tmp_path):
    """The streaming path is only safe if the blocks are exactly the forecast:
    same rows, same order, same values as the materialized form."""
    _out, (train, val, test) = _built_splits(tmp_path)
    targets = discover_targets(test)
    for name in ("naive", "fade"):
        whole = run_baseline(name, train, test, val_df=val, targets=targets)
        streamed = pd.concat(
            iter_baseline_blocks(name, train, test, val_df=val, targets=targets),
            ignore_index=True)
        pd.testing.assert_frame_equal(whole, streamed)
        # every (target, horizon) cell of the test split, once
        assert len(whole) == len(test) * len(target_columns(test, targets))


def test_run_baselines_writes_the_same_file_the_in_memory_path_would(tmp_path):
    """`baselines` writes via the streamed row-group writer; the file must match
    what write_forecast(run_baseline(...)) produced before (I-3)."""
    out, (train, val, test) = _built_splits(tmp_path)
    suffix = "pf_full__t"
    written = run_baselines(tmp_path / "processed", suffix, tmp_path / "fc",
                            which=["naive", "fade"], verbose=False)
    targets = discover_targets(test)
    for name, path in written.items():
        got = read_forecast(path, validate=False)
        want = run_baseline(name, train, test, val_df=val, targets=targets)
        assert len(got) == len(want)
        np.testing.assert_array_equal(got["firm"].to_numpy(), want["firm"].to_numpy())
        np.testing.assert_array_equal(got["target"].to_numpy(), want["target"].to_numpy())
        np.testing.assert_array_equal(got["horizon"].to_numpy(), want["horizon"].to_numpy())
        np.testing.assert_array_equal(got["prediction"].to_numpy(),
                                      want["prediction"].to_numpy().astype(np.float32))


def test_naive_alone_still_forecasts_every_truth_cell(tmp_path):
    """`--which naive` projects the 1,560 truth columns out of the read, but must
    still predict all of them: what to forecast comes from the file's schema, not
    from the columns that were loaded."""
    _out, (_train, _val, test) = _built_splits(tmp_path)
    written = run_baselines(tmp_path / "processed", "pf_full__t", tmp_path / "fc",
                            which=["naive"], verbose=False)
    got = read_forecast(written["naive"], validate=False)
    targets = discover_targets(test)
    assert len(got) == len(test) * len(target_columns(test, targets))
    assert sorted(got["target"].unique()) == sorted(targets)
    assert sorted(got["horizon"].unique()) == sorted(
        {int(c.rsplit("_t", 1)[-1]) for c in target_columns(test, targets)})


def test_baseline_column_projection_reads_only_what_it_needs():
    """The three canonical splits are 2,547 columns / ~12 GB together. `naive`
    reads one column per target, `fade` adds the 20 horizons; only the linear
    family needs the feature matrix. Reading everything regardless is what put
    `baselines --which naive,fade` over the memory ceiling (I-3)."""
    from proforma20q.baselines.run import _columns_for

    cols = (["firm", "origin", "scale_level_0"]
            + [f"niq_level_{i}" for i in range(4)]
            + [f"niq_yoy_{i}" for i in range(8)]
            + [f"niq_t{h}" for h in range(1, 21)]
            + [f"indff48_{i}" for i in range(48)])
    targets = ["niq"]

    naive = _columns_for(["naive"], cols, targets)
    assert naive == ["firm", "origin", "niq_level_0"]

    fade = _columns_for(["fade"], cols, targets)
    assert set(fade) == {"firm", "origin", "niq_level_0"} | {f"niq_t{h}" for h in range(1, 21)}

    full = _columns_for(["linear"], cols, targets)
    assert set(full) == set(cols)
    assert full == [c for c in cols if c in set(full)]   # original column order


def test_elasticnet_and_linear_run(tmp_path):
    raw = synthetic_raw(n_firms=20)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)
    out = build(raw_path, tmp_path / "processed", dataset_tag="t", which=("tabular",), verbose=False)
    train = load_tabular(out["tabular_train"])
    val = load_tabular(out["tabular_val"])
    test = load_tabular(out["tabular_test"])
    targets = discover_targets(test)[:3]  # keep the CV cheap
    fcs = {}
    for name in ("linear", "elasticnet"):
        fcs[name] = run_baseline(name, train, test, val_df=val, targets=targets, verbose=False)
    res = evaluate_forecasts(fcs, test, verbose=False)
    assert res.invariant_ok
    # linear/elasticnet should beat the RW anchor on average (they see the features).
    lb = res.leaderboard()
    assert lb["r2"].notna().all()
