"""Tests for scripts/build_full_sample_mask.py (mask build + keys decode)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import hashlib
import re

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


def test_keys_mask_survives_vintage_drift_and_bits_mask_does_not(tmp_path):
    """The point of I-6. A rebuild one test row short of the canonical grid
    cannot use the grid-aligned mask at all -- it is a bitmap over an exact row
    set -- while the keys mask matches by value and simply intersects."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=25, n_q=5, seed=4)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[::3] = True
    keys = bmask.mask_to_keys(mask, grid, truth_path)

    # a later vintage: one test row disappears
    drifted = truth.iloc[1:].reset_index(drop=True)
    fc = forecast_from_truth(drifted, noise=0.3, seed=2)

    with pytest.raises(ValueError, match="VINTAGE DRIFT"):
        evaluate_forecasts({"m": fc}, drifted, sample_mask=np.packbits(mask), verbose=False)

    res = evaluate_forecasts({"m": fc}, drifted, sample_mask=keys, verbose=False)
    assert res.n_common > 0
    # exactly the masked cells that still exist in the drifted grid
    dropped = truth["firm"].iloc[0], pd.Timestamp(truth["origin"].iloc[0])
    survivors = keys[~((keys["firm"] == dropped[0]) &
                       (pd.to_datetime(keys["origin"]) == dropped[1]))]
    assert res.n_common == len(survivors)


def test_bits_to_keys_conversion_needs_only_the_truth(tmp_path):
    """`--from-bits` produces the portable mask from the already-published bit
    array plus the canonical tabular_test -- no forecast, which is the artifact
    deferred to publication."""
    truth = synthetic_truth(n_firms=12, n_q=4, seed=11)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[[0, 5, 17, grid.n_cells - 1]] = True
    bits_path = tmp_path / "bits.npy"
    np.save(bits_path, np.packbits(mask))

    rc = bmask.main(["--from-bits", str(bits_path), "--truth", str(truth_path),
                     "--out", str(tmp_path / "out"), "--expect", "4"])
    assert rc == 0
    keys = pd.read_parquet(tmp_path / "out" / "full_sample_mask_keys.parquet")
    assert len(keys) == 4
    pd.testing.assert_frame_equal(
        keys.reset_index(drop=True),
        bmask.mask_to_keys(mask, grid, truth_path).reset_index(drop=True))

    # a mask built against a different grid is refused, not silently sliced
    other = synthetic_truth(n_firms=13, n_q=4, seed=11)
    other_path = tmp_path / "other.parquet"
    other.to_parquet(other_path, index=False)
    with pytest.raises(SystemExit, match="only converts against the SAME"):
        bmask.main(["--from-bits", str(bits_path), "--truth", str(other_path),
                    "--out", str(tmp_path / "out2")])


def test_download_artifacts_md5_guard_and_pins(tmp_path):
    dl = _load_script("download_artifacts.py")
    # md5sum matches hashlib
    p = tmp_path / "x.bin"
    p.write_bytes(b"proforma-20q")
    assert dl.md5sum(p) == hashlib.md5(b"proforma-20q").hexdigest()
    # every pin is a full 32-hex md5 -- catches placeholders AND the
    # transcription bug class (an ellipsized or truncated paste)
    assert all(re.fullmatch(r"[0-9a-f]{32}", v) for v in dl.ARTIFACTS.values())
    # the release-placeholder guard fires until ZENODO_RECORD is set
    with pytest.raises(SystemExit):
        dl.main(["--only", "full_sample_mask_bits.npy", "--out", str(tmp_path)])


def test_download_artifacts_only_cannot_drop_the_sidecar():
    """A density forecast without its {stem}.nll.json is silently scored as
    Gaussian, so --only must pull the sidecar along with its parquet."""
    dl = _load_script("download_artifacts.py")
    lap = "forma_lap05_fgrid__pf_full__test__predictions.parquet"
    side = "forma_lap05_fgrid__pf_full__test__predictions.nll.json"
    assert dl._with_sidecars([lap]) == [lap, side]
    assert dl._with_sidecars([lap, side]) == [lap, side]  # no duplicate
    assert dl._with_sidecars(["full_sample_mask_bits.npy"]) == ["full_sample_mask_bits.npy"]
    # the full-manifest default keeps every pinned artifact exactly once
    assert sorted(dl._with_sidecars(dl.ARTIFACTS)) == sorted(dl.ARTIFACTS)


def test_readme_artifact_digests_match_script_pins():
    """The README artifact tables duplicate 8-hex digest prefixes by hand; a
    hand copy is exactly how the superseded run-1 pin went stale (gh#15).
    Update policy: change scripts/download_artifacts.py first, then make the
    README rows agree -- this test fails on any drift between the two."""
    dl = _load_script("download_artifacts.py")
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"\|\s*\[?`([A-Za-z0-9_./]+)`\]?[^|]*\|[^|]*\|\s*`([0-9a-f]{8})…`", readme)
    prefixes: dict[str, str] = {}
    for path, pref in rows:
        name = path.rsplit("/", 1)[-1]
        assert prefixes.get(name, pref) == pref, f"README lists {name} with conflicting digests"
        prefixes[name] = pref
    for name, md5 in dl.ARTIFACTS.items():
        assert name in prefixes, f"{name} is pinned in download_artifacts.py but absent from the README tables"
        assert md5.startswith(prefixes[name]), \
            f"README digest `{prefixes[name]}…` for {name} does not match the pinned md5 {md5}"
