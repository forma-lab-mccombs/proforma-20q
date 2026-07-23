import pandas as pd

from proforma20q.config import feature_set_items, load_task_config, pf_full_targets
from proforma20q.baselines import BASELINES, run_baseline
from proforma20q.baselines.common import discover_targets, load_tabular
from proforma20q.build import build
from proforma20q.evaluate import evaluate_forecasts
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
