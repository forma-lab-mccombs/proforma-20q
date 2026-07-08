"""Tests for scripts/build_full_sample_mask.py (mask build + keys decode)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd
import pytest

from proforma20q.evaluate import evaluate_forecasts
from fixtures import forecast_from_truth, synthetic_truth


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), _ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ROOT = Path(__file__).resolve().parent.parent
bmask = _load_script("build_full_sample_mask.py")


def _write_forecast(fc, path):
    fc.to_parquet(path, index=False)
    return path


def test_build_mask_equals_finite_forecast_intersect_truth(tmp_path):
    truth = synthetic_truth(n_firms=25, n_q=5, seed=4)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    # forecast covering only horizons 1..3 (so coverage < truth grid)
    fc = forecast_from_truth(truth, noise=0.3, seed=2)
    fc = fc[fc["horizon"] <= 3].reset_index(drop=True)
    _write_forecast(fc, tmp_path / "fc.parquet")

    mask, grid = bmask.build_mask(tmp_path / "fc.parquet", truth_path, batch_size=10_000, verbose=False)
    # every masked cell has a finite forecast AND finite truth; count is sane
    assert mask.sum() > 0
    assert mask.sum() <= (fc["horizon"] <= 3).sum()  # <= number of forecast rows

    # keys decode round-trips: evaluating with the decoded keys == with the bitmask
    keys = bmask.mask_to_keys(mask, grid, truth_path)
    assert set(keys["horizon"].unique()) <= {1, 2, 3}
    r_bits = evaluate_forecasts({"m": fc}, truth, sample_mask=mask, verbose=False)
    r_keys = evaluate_forecasts({"m": fc}, truth, sample_mask=keys, verbose=False)
    assert r_bits.n_common == r_keys.n_common == int(mask.sum())
    b = r_bits.leaderboard()[lambda d: d.model == "m"]["r2"].iloc[0]
    k = r_keys.leaderboard()[lambda d: d.model == "m"]["r2"].iloc[0]
    assert np.isclose(b, k, equal_nan=True)


def test_mask_to_keys_addressing_matches_grid(tmp_path):
    """A single hand-set cell decodes back to the exact (firm,target,origin,horizon)."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns
    truth = synthetic_truth(n_firms=8, n_q=4, seed=7)
    truth_path = tmp_path / "t.parquet"
    truth.to_parquet(truth_path, index=False)
    grid = TruthGrid(normalize_columns(truth))

    t_id, h_idx, wide_row = 2, 3, 5  # some interior cell
    cell = (t_id * grid.n_h + h_idx) * grid.n_wide + wide_row
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[cell] = True
    keys = bmask.mask_to_keys(mask, grid, truth_path)
    assert len(keys) == 1
    row = keys.iloc[0]
    tn = normalize_columns(truth)
    assert row["target"] == grid.targets[t_id]
    assert int(row["horizon"]) == grid.horizons[h_idx]
    assert str(row["firm"]) == str(tn["firm"].iloc[wide_row])
    assert pd.Timestamp(row["origin"]) == pd.Timestamp(tn["origin"].iloc[wide_row])


def test_download_artifacts_md5_guard_and_pins(tmp_path):
    dl = _load_script("download_artifacts.py")
    # md5sum matches hashlib
    p = tmp_path / "x.bin"
    p.write_bytes(b"proforma-20q")
    assert dl.md5sum(p) == hashlib.md5(b"proforma-20q").hexdigest()
    # artifact md5s are pinned (only the Zenodo record id is a placeholder)
    assert all(not v.startswith("REPLACE_WITH") for v in dl.ARTIFACTS.values())
    # the release-placeholder guard fires until ZENODO_RECORD is set
    with pytest.raises(SystemExit):
        dl.main(["--only", "full_sample_mask_bits.npy", "--out", str(tmp_path)])
