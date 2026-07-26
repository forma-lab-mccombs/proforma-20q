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


def test_short_history_firm_does_not_crash_build(tmp_path):
    """A firm with fewer quarters than the max feature lag (11) must not crash the
    build. The old shift `raw_vals[:n-lag]` trimmed from the wrong end once the lag
    exceeded a firm's history (n-lag < 0), raising a broadcast error during panel
    construction; the pandas-`shift` port fixes it. Late-entry firms hit this in
    the real R13 test split. (Deep-lag values are later imputed, so this is a
    crash-regression guard, not a NaN assertion.)"""
    raw = synthetic_raw()
    short = "00007"
    # 3-quarter history landing in the 2010+ test window (index 56 ~ 2010Q1), so
    # it survives the min_lookback Condition 0 and flows through construction.
    keep = raw[raw["gvkey"] == short].sort_values("datadate").iloc[56:59]
    raw = pd.concat([raw[raw["gvkey"] != short], keep], ignore_index=True)
    raw_path = tmp_path / "compustat_with_permno.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)

    out = build(raw_path, tmp_path / "proc", dataset_tag="short", which=("tabular",), verbose=False)
    # Completed without the broadcast crash, and the short-history firm flows
    # through into the built output.
    seen = any((load_tabular(out[s])["firm"].astype(str) == short).any()
               for s in ("tabular_train", "tabular_val", "tabular_test"))
    assert seen


def test_imputation_column_set_includes_scale_excludes_targets_and_dummies(built):
    """The Local-XS imputation matrix must include `scale_level_0` (matches the
    research repo -- it is never filled but shifts the factor loadings) and must
    exclude industry dummies and future-target `_t{h}` columns. Mirrors the
    `impute_cols` selector in build_tabular. Guards fix #5."""
    test = load_tabular(built["tabular_test"])
    impute_cols = [c for c in test.columns if "_level_" in c or "_yoy_" in c]
    assert "scale_level_0" in impute_cols
    assert not any(c.startswith("indff48_") for c in impute_cols)
    assert not any("_t" in c and c.rsplit("_t", 1)[-1].isdigit() for c in impute_cols)


def test_yoy_clamp_is_wider_than_levels_by_1p4142():
    """YoY clamps to +/- max_z*1.4142 (literal, not sqrt(2)); levels clamp to
    +/- max_z. Guards fixes #2 (differences UNCLIPPED normals) and #4."""
    from proforma20q.build import _YOY_Z_FACTOR, _normalize
    assert _YOY_Z_FACTOR == 1.4142  # exact literal, not np.sqrt(2)

    # Two unclipped normals, one beyond +/-max_z: clipping BEFORE differencing
    # (the bug) gives a different result than differencing THEN clipping.
    scale = np.array([1.0]); k = 1.0
    a = _normalize(np.array([np.sinh(9.0)]), scale, k, np.array([0.0]), np.array([1.0]))
    b = _normalize(np.array([np.sinh(1.0)]), scale, k, np.array([0.0]), np.array([1.0]))
    assert a[0] > 6.0  # a is beyond the level clamp
    max_z = 6.0
    yoy_correct = np.clip(a - b, -max_z * 1.4142, max_z * 1.4142)[0]
    yoy_buggy = np.clip(np.clip(a, -max_z, max_z) - np.clip(b, -max_z, max_z),
                        -max_z * 1.4142, max_z * 1.4142)[0]
    assert not np.isclose(yoy_correct, yoy_buggy)  # the two paths genuinely differ
    assert np.isclose(yoy_correct, np.clip(8.0, -max_z * 1.4142, max_z * 1.4142))  # 9-1=8


def _prepped_synthetic(raw=None):
    """Run the orchestrator's prep steps so the tabular builder can be called
    directly (what `build()` hands to `prepare_panel`)."""
    from proforma20q.build import (add_computed_features, compute_scale,
                                   convert_ytd_to_quarterly, firm_ff48_map)
    from proforma20q.config import feature_set_items, load_task_config
    task = load_task_config()
    raw = synthetic_raw() if raw is None else raw
    raw = raw.rename(columns={"gvkey": "firm_id", "datadate": "quarter", "naicsh": "naics"})
    raw["firm_id"] = raw["firm_id"].astype(str)
    firm_ind = firm_ff48_map(raw)
    raw["quarter"] = pd.to_datetime(raw["quarter"]).dt.to_period("Q").dt.end_time
    raw = convert_ytd_to_quarterly(raw)
    items = feature_set_items(task["benchmark"]["feature_set"])
    raw = add_computed_features(raw, items)
    raw["scale"] = compute_scale(raw, task["scaling"])
    targets = [it for it in items if it in raw.columns]
    return raw, items, targets, firm_ind, task


def test_vectorized_panel_matches_per_firm_expansion():
    """`prepare_panel` expands every firm at once; it must agree cell-for-cell
    with applying the per-firm `_contiguous_panel` reference form. Guards I-1:
    the pre-allocated builder is only correct if positional shifts still equal
    calendar lags for every firm, including firms with quarter gaps."""
    from proforma20q.build import _contiguous_panel, prepare_panel

    raw, items, _t, _fi, _task = _prepped_synthetic()
    # Punch gaps into two firms so the expansion has real work to do.
    drop = ((raw["firm_id"] == "00003") & (raw["quarter"].dt.year == 2004)) | \
           ((raw["firm_id"] == "00007") & (raw["quarter"].dt.quarter == 2))
    raw = raw[~drop].reset_index(drop=True)

    panel = prepare_panel(raw, items)
    expected = pd.concat(
        [_contiguous_panel(g, "quarter") for _f, g in
         raw.sort_values(["firm_id", "quarter"]).groupby("firm_id", sort=False)],
        ignore_index=True)

    assert len(panel) == len(expected)
    assert (panel.quarter == expected["quarter"].to_numpy()).all()
    # firm_id is NaN on gap-filled rows of the reference form; compare the ids we
    # reconstruct against the per-firm block layout instead.
    for col in ("scale", "niq", "revtq", "atq"):
        np.testing.assert_array_equal(panel.values[col],
                                      expected[col].to_numpy(float), err_msg=col)
    # positional shift == calendar lag: 4 rows back is exactly one year back
    back4 = np.roll(panel.qord, 4)
    inner = panel.pos_in_firm >= 4
    assert (panel.qord[inner] - back4[inner] == 4).all()


def test_split_builder_matches_full_frame_builder():
    """`build_tabular_splits` (used by `build()`) must be bit-identical to
    `split_tabular(build_tabular(...))`. The split path exists only to keep the
    unsplit 12 GB matrix from ever being materialized (I-1), so any divergence
    between the two is a silent redefinition of the artifacts."""
    from proforma20q.build import (build_tabular, build_tabular_splits,
                                   create_regularization_stats, prepare_panel,
                                   split_tabular)

    raw, items, targets, firm_ind, task = _prepped_synthetic()
    fe, splits = task["feature_engineering"], task["splits"]
    reg = create_regularization_stats(
        raw, items, fe, pd.Timestamp(f"{splits['train_end_year']}-12-31"))
    kw = dict(tabular_industry_fe=task["industry"]["tabular_industry_fe"],
              present_in_q0=task["targets"]["present_in_q0"])

    whole = build_tabular(raw, reg, items, targets, fe, firm_ind, **kw)
    expected = dict(zip(("train", "val", "test"),
                        split_tabular(whole, splits["train_end_year"],
                                      splits["val_end_year"], targets,
                                      fe["forecast_horizon"])))
    got = dict(build_tabular_splits(prepare_panel(raw, items), reg, targets, fe,
                                    firm_ind, splits, **kw))

    for name in ("train", "val", "test"):
        a, b = expected[name], got[name]
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape and len(a) > 0
        assert (a["firm_id"].to_numpy() == b["firm_id"].to_numpy()).all()
        assert (a["quarter"].to_numpy() == b["quarter"].to_numpy()).all()
        av, bv = a.iloc[:, 2:].to_numpy(), b.iloc[:, 2:].to_numpy()
        np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
        both = ~np.isnan(av)
        np.testing.assert_array_equal(av[both], bv[both])  # bit-identical, not allclose


def test_build_ignores_unconsumed_raw_columns(tmp_path):
    """`build` reads only the columns it consumes (86 of a `SELECT f.*` panel's
    655). Extra columns must change nothing about the output -- and must not be
    loaded, which is what keeps the raw panel out of the ~25 GB range (I-2)."""
    from proforma20q.build import required_raw_columns

    raw = synthetic_raw(n_firms=8)
    lean = tmp_path / "lean.parquet"
    raw.to_parquet(lean, engine="fastparquet", index=False)

    fat_df = raw.copy()
    for i in range(40):  # columns comp.fundq ships and the benchmark never reads
        fat_df[f"junk{i}"] = np.arange(len(raw), dtype=float)
    fat = tmp_path / "fat.parquet"
    fat_df.to_parquet(fat, engine="fastparquet", index=False)

    a = build(lean, tmp_path / "a", dataset_tag="a", which=("tabular",), verbose=False)
    b = build(fat, tmp_path / "b", dataset_tag="b", which=("tabular",), verbose=False)
    assert not any(c.startswith("junk") for c in required_raw_columns())
    for split in ("tabular_train", "tabular_val", "tabular_test"):
        pd.testing.assert_frame_equal(load_tabular(a[split]), load_tabular(b[split]))


def test_prepare_panel_rejects_duplicate_firm_quarter():
    """The per-firm `reindex` form raised on duplicate keys; the vectorized form
    would silently keep whichever row was scattered last, so it must refuse too."""
    from proforma20q.build import prepare_panel

    raw, items, _t, _fi, _task = _prepped_synthetic()
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_panel(dup, items)


def test_prepare_panel_rejects_missing_keys():
    """A null key is the one input that corrupts silently: `pd.factorize` codes
    it -1, which indexes the LAST firm's block from the end and scatters the row
    into another firm's panel; a NaT quarter maps to a sentinel ordinal ~8,000
    quarters before the data and inflates that firm's contiguous range. The old
    per-firm groupby/reindex dropped both. Refusing is the substitute."""
    from proforma20q.build import prepare_panel

    raw, items, _t, _fi, _task = _prepped_synthetic()

    no_firm = raw.copy()
    no_firm.loc[no_firm.index[0], "firm_id"] = None
    with pytest.raises(ValueError, match="missing firm_id"):
        prepare_panel(no_firm, items)

    no_quarter = raw.copy()
    no_quarter.loc[no_quarter.index[0], "quarter"] = pd.NaT
    with pytest.raises(ValueError, match="missing quarter"):
        prepare_panel(no_quarter, items)


def test_prepare_panel_tolerates_a_repeated_item():
    """`items` with a duplicate used to emit each block twice, with only the
    second copy written -- the first stayed all-NaN."""
    from proforma20q.build import prepare_panel

    raw, items, _t, _fi, _task = _prepped_synthetic()
    panel = prepare_panel(raw, ["niq", "revtq", "niq"])
    assert panel.items == ["niq", "revtq"]


def test_build_reads_a_panel_that_already_uses_the_renamed_keys(tmp_path):
    """`build --raw <panel>` accepts any Compustat-shaped parquet, including one
    already carrying firm_id/quarter instead of gvkey/datadate. The column
    projection must ask for whichever spelling the file has."""
    raw = synthetic_raw(n_firms=8).rename(
        columns={"gvkey": "firm_id", "datadate": "quarter", "naicsh": "naics"})
    p = tmp_path / "renamed.parquet"
    raw.to_parquet(p, engine="fastparquet", index=False)
    out = build(p, tmp_path / "proc", dataset_tag="rn", which=("tabular",), verbose=False)
    assert len(load_tabular(out["tabular_test"])) > 0


def test_build_refuses_to_write_an_empty_build(tmp_path):
    """A panel where nothing survives the keep-mask must fail loudly, not write
    three empty parquets and report success."""
    raw = synthetic_raw(n_firms=8)
    for col in ("atq", "ltq", "seqq"):
        raw[col] = np.nan          # no usable scale anywhere
    p = tmp_path / "noscale.parquet"
    raw.to_parquet(p, engine="fastparquet", index=False)
    with pytest.raises(ValueError, match="keep-mask"):
        build(p, tmp_path / "proc", dataset_tag="empty", which=("tabular",), verbose=False)


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


def test_build_persists_the_tuple_id_maps(tmp_path):
    """The tuple view stores integer ids; the submission schema needs the gvkey
    string and the item name. The maps are the only bridge, so a build that
    drops them cannot yield a valid submission from the tuple view at all."""
    from proforma20q.build import read_id_maps

    raw = synthetic_raw(n_firms=9)
    raw_path = tmp_path / "raw.parquet"
    raw.to_parquet(raw_path, engine="fastparquet", index=False)
    out = build(raw_path, tmp_path / "proc", dataset_tag="ids", which=("tuple",),
                verbose=False)
    for key in ("firm_id_map", "account_id_map", "industry_id_map"):
        assert key in out and out[key].exists()

    maps = read_id_maps(tmp_path / "proc", "pf_full__ids")
    tup = pd.read_parquet(out["tuple_test"], engine="fastparquet")

    # every integer id in the artifact resolves through the maps
    firm = maps["firm_id_map"].set_index("firm_id_int")["firm_id"]
    acct = maps["account_id_map"].set_index("account_id")["account_name"]
    assert set(tup["firm_id"]).issubset(set(firm.index))
    assert set(tup["account_id"]).issubset(set(acct.index))
    assert set(tup["industry_id"]).issubset(set(maps["industry_id_map"]["industry_id"]))

    # ... and back to the identifiers the submission schema requires
    gvkeys = set(raw["gvkey"].astype(str))
    assert set(firm.loc[sorted(set(tup["firm_id"]))]).issubset(gvkeys)
    assert "niq" in set(acct)


def test_id_maps_preserve_zero_padded_gvkeys(tmp_path):
    """CSV has no types: a default read turns the gvkey "001045" into 1045, which
    joins to nothing in the truth file. The bundled reader must not."""
    from proforma20q.build import read_id_maps, write_id_maps

    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    firm_map = {"001045": 0, "001004": 1, "0012345": 2}
    write_id_maps(out_dir, "pf_full__z", firm_map, {"niq": 0, "atq": 1})

    naive = pd.read_csv(out_dir / "firm_id_map__pf_full__z.csv")
    assert naive["firm_id"].dtype != object     # the trap: silently numeric

    maps = read_id_maps(out_dir, "pf_full__z")
    assert set(maps["firm_id_map"]["firm_id"]) == set(firm_map)
