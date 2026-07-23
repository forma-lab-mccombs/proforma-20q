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


def test_sample_mask_grid_aligned_bitmask_matches_keys(tmp_path):
    """A grid-aligned bit array, its packbits form, an on-disk .npy, and the
    equivalent keys mask all restrict to the same sample AND the same R²."""
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
    npy = tmp_path / "mask.npy"
    np.save(npy, np.packbits(bits))

    def run(m):
        r = evaluate_forecasts({"m": fc}, truth, sample_mask=m, verbose=False)
        return r.n_common, float(r.leaderboard()[lambda d: d.model == "m"]["r2"].iloc[0])

    keys_res = run(keys)
    for m in (bits, np.packbits(bits), str(npy)):
        n, r2 = run(m)
        assert n == keys_res[0]
        assert np.isclose(r2, keys_res[1], equal_nan=True)


def test_sample_mask_wrong_grid_size_errors():
    """A grid-aligned mask sized for a different grid is rejected, not silently
    truncated and misapplied."""
    truth = synthetic_truth(n_firms=10, n_q=4, seed=2)
    fc = forecast_from_truth(truth, noise=0.5, seed=1)
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns
    n_cells = TruthGrid(normalize_columns(truth)).n_cells
    for bad in (np.ones(n_cells + 100, dtype=bool),          # oversized bool
                np.ones(n_cells - 5, dtype=bool),            # undersized bool
                np.packbits(np.ones(n_cells + 800, dtype=bool))):  # packbits of wrong grid
        with pytest.raises(ValueError, match="grid"):
            evaluate_forecasts({"m": fc}, truth, sample_mask=bad, verbose=False)


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


def test_student_t_crps_matches_quadrature():
    """The location-scale student_t CRPS closed form must agree with the integral
    definition CRPS(F, y) = ∫ (F(x) − 1{x≥y})² dx to ≤ 1e-4, across several
    (df, sigma, residual) combos -- the mandatory numerical check before shipping
    the closed form (df ∈ {1.5, 3, 5, 30})."""
    from scipy import integrate, stats

    from proforma20q.evaluate import crps_per_entry

    def crps_quad(res, sigma, nu):
        # predictive t centered at the forecast (loc=res) scored against y=0, so
        # the standardized residual is -res/sigma (CRPS is symmetric).
        dist = stats.t(df=nu, loc=res, scale=sigma)
        integrand = lambda x: (dist.cdf(x) - (1.0 if x >= 0.0 else 0.0)) ** 2  # noqa: E731
        lo, hi = res - 400.0 * sigma, res + 400.0 * sigma
        val, _ = integrate.quad(integrand, lo, hi, limit=500, points=[0.0])
        return val

    worst = 0.0
    for nu in (1.5, 3.0, 5.0, 30.0):
        for sigma in (0.5, 1.0, 2.3):
            for res in (0.0, 0.7, -1.9, 5.0):
                closed = float(crps_per_entry(np.array([res]), np.array([sigma]),
                                              "student_t", nu)[0])
                worst = max(worst, abs(closed - crps_quad(res, sigma, nu)))
    assert worst <= 1e-4, f"worst student_t CRPS error {worst:.2e} exceeds 1e-4"


def test_student_t_crps_below_df_one_is_nan():
    """nu <= 1 (no finite CRPS) drops out as NaN instead of raising or lying."""
    from proforma20q.evaluate import crps_per_entry

    out = crps_per_entry(np.array([0.5, -1.0]), np.array([1.0, 1.0]), "student_t", 1.0)
    assert np.isnan(out).all()


def test_student_t_sidecar_produces_finite_crps():
    """A student_t forecast (declared as a .nll.json sidecar would, here via the
    families override) now yields a finite CRPS, not NaN."""
    truth = synthetic_truth(n_firms=40, n_q=6, seed=3)
    fc = forecast_from_truth(truth, noise=1.0, sigma=1.0, seed=4)
    res = evaluate_forecasts({"t": fc}, truth,
                             families={"t": ("student_t", 5.0)}, verbose=False)
    row = res.leaderboard()[lambda d: d.model == "t"].iloc[0]
    assert np.isfinite(row["nll"])
    assert np.isfinite(row["crps"]) and row["crps"] > 0


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
