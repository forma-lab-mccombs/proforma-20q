import math

import numpy as np
import pytest

from proforma20q.evaluate import evaluate_forecasts
from fixtures import forecast_from_truth, synthetic_truth


def test_perfect_forecast_scores_r2_one():
    truth = synthetic_truth()
    perfect = forecast_from_truth(truth, noise=0.0)
    res = evaluate_forecasts({"perfect": perfect}, truth, verbose=False)
    lb = res.leaderboard("r2")
    row = lb[lb.model == "perfect"].iloc[0]
    assert row["r2"] > 0.9999
    assert row["mae"] < 1e-6
    assert row["mse"] < 1e-6
    assert res.invariant_ok


def test_common_sample_invariant_and_size():
    truth = synthetic_truth(n_firms=20, n_q=4)
    a = forecast_from_truth(truth, noise=0.5, seed=1)
    b = forecast_from_truth(truth, noise=1.0, seed=2)
    res = evaluate_forecasts({"a": a, "b": b}, truth, verbose=False)
    assert res.invariant_ok
    # every cell present in both -> common sample == full grid
    n_cells = 20 * 4 * 3 * 5
    assert res.n_common == n_cells


def test_sample_mask_restricts_to_pooled_subset():
    """--sample-mask restricts scoring to a fixed pooled cell set; the scored
    sample is mask INTERSECT the all-models common sample."""
    truth = synthetic_truth(n_firms=40, n_q=6, seed=5)
    fc = forecast_from_truth(truth, noise=0.4, seed=3)
    full = evaluate_forecasts({"m": fc}, truth, verbose=False)

    # Mask = only the horizon-1 keys.
    mask = fc[fc["horizon"] == 1][["firm", "target", "origin", "horizon"]]
    masked = evaluate_forecasts({"m": fc}, truth, sample_mask=mask, verbose=False)

    assert masked.n_common < full.n_common
    bh = full.metrics["by_horizon"]
    h1 = int(bh[bh["horizon"] == 1]["n_complete"].iloc[0])
    assert masked.n_common == h1  # exactly the horizon-1 common cells
    # only horizon 1 retains cells; the masked-out horizons drop to n_complete 0
    mbh = masked.metrics["by_horizon"].set_index("horizon")["n_complete"]
    assert mbh[1] == h1
    assert (mbh.drop(1) == 0).all()


def test_sample_mask_grid_aligned_bitmask_matches_keys():
    """A grid-aligned bit array (and its packbits form) restricts identically to
    the equivalent keys mask."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns
    truth = synthetic_truth(n_firms=20, n_q=5, seed=6)
    fc = forecast_from_truth(truth, noise=0.4, seed=1)
    grid = TruthGrid(normalize_columns(truth))

    bits = np.zeros(grid.n_cells, dtype=bool)
    for t_id in range(len(grid.targets)):
        start = (t_id * grid.n_h + 0) * grid.n_wide  # h_idx 0 == smallest horizon (1)
        bits[start:start + grid.n_wide] = True

    keys = fc[fc["horizon"] == 1][["firm", "target", "origin", "horizon"]]
    n_keys = evaluate_forecasts({"m": fc}, truth, sample_mask=keys, verbose=False).n_common
    n_bits = evaluate_forecasts({"m": fc}, truth, sample_mask=bits, verbose=False).n_common
    n_packed = evaluate_forecasts({"m": fc}, truth, sample_mask=np.packbits(bits), verbose=False).n_common
    assert n_bits == n_keys == n_packed


def test_sample_mask_off_grid_keys_ignored():
    """Keys not on the truth grid are dropped from the mask, not errored: a mask of
    (all real keys + some off-grid firms) scores the same as no mask at all."""
    import pandas as pd
    truth = synthetic_truth(n_firms=15, n_q=4, seed=8)
    fc = forecast_from_truth(truth, noise=0.5, seed=2)
    mask = fc[["firm", "target", "origin", "horizon"]].copy()
    bogus = mask.head(3).copy()
    bogus["firm"] = "999999"  # numeric but not among the 15 truth firms -> off-grid
    mask = pd.concat([mask, bogus], ignore_index=True)
    res = evaluate_forecasts({"m": fc}, truth, sample_mask=mask, verbose=False)
    assert res.n_common == evaluate_forecasts({"m": fc}, truth, verbose=False).n_common


def test_gaussian_probabilistic_metrics():
    truth = synthetic_truth(n_firms=60, n_q=8, seed=3)
    fc = forecast_from_truth(truth, noise=1.0, sigma=1.0, seed=4)
    res = evaluate_forecasts({"g": fc}, truth, verbose=False)
    row = res.leaderboard()[lambda d: d.model == "g"].iloc[0]
    # perfectly-calibrated unit Gaussian: NLL -> 0.5*ln(2pi)+0.5, z2 -> 1, cover95 -> 0.95
    assert abs(row["nll"] - (0.5 * math.log(2 * math.pi) + 0.5)) < 0.05
    assert abs(row["z2"] - 1.0) < 0.05
    assert abs(row["cover95"] - 0.95) < 0.02
    assert row["crps"] > 0


def test_missing_forecast_raises_without_allow_missing():
    truth = synthetic_truth(n_firms=5, n_q=2)
    good = forecast_from_truth(truth, noise=0.1)
    with pytest.raises(RuntimeError):
        evaluate_forecasts({"good": good, "bad": "does_not_exist.parquet"}, truth, verbose=False)


def test_shared_r2_denominator_is_model_independent():
    truth = synthetic_truth(n_firms=15, n_q=4)
    a = forecast_from_truth(truth, noise=0.3, seed=1)
    b = forecast_from_truth(truth, noise=0.9, seed=2)
    res_both = evaluate_forecasts({"a": a, "b": b}, truth, verbose=False)
    res_a = evaluate_forecasts({"a": a}, truth, verbose=False)
    ra_both = res_both.leaderboard()[lambda d: d.model == "a"]["r2"].iloc[0]
    ra_solo = res_a.leaderboard()[lambda d: d.model == "a"]["r2"].iloc[0]
    # a's rows are complete in both pools, so its R2 is identical either way.
    assert abs(ra_both - ra_solo) < 1e-9
