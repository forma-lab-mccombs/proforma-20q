"""Tests for scripts/build_full_sample_mask.py (mask build + keys decode)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import hashlib
import re
import subprocess

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


def test_grid_rows_makes_the_bits_mask_survive_vintage_drift(tmp_path):
    """The counterpart to the test above: with the published canonical row index,
    the compact bit array *does* apply to a drifted rebuild, and lands on exactly
    the cells the portable keys form selects."""
    from proforma20q.truth_grid import TruthGrid, canon_origin
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=25, n_q=5, seed=4)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[::3] = True
    keys = bmask.mask_to_keys(mask, grid, truth_path)

    rows_path = tmp_path / "full_sample_grid_rows.parquet"
    n_rows = bmask.write_grid_rows(grid, rows_path)
    assert n_rows == grid.n_wide
    gr = pd.read_parquet(rows_path)
    assert list(gr.columns) == ["grid_row", "firm", "origin"]
    # Stored as TEXT, not a timestamp -- that is what keeps the us/ns precision
    # trap out of the index. Assert the property, not the dtype spelling: pandas 3
    # reads these back as StringDtype where 2.x gives object.
    assert not pd.api.types.is_datetime64_any_dtype(gr["origin"])
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)) for v in gr["origin"].head(10))
    assert canon_origin(gr["origin"]).equals(canon_origin(normalize_columns(truth)["origin"]))
    assert gr["grid_row"].tolist() == list(range(grid.n_wide))

    # a later vintage: one test row disappears
    drifted = truth.iloc[1:].reset_index(drop=True)
    fc = forecast_from_truth(drifted, noise=0.3, seed=2)

    # without the index the bitmap is unusable ...
    with pytest.raises(ValueError, match="VINTAGE DRIFT"):
        evaluate_forecasts({"m": fc}, drifted, sample_mask=np.packbits(mask), verbose=False)

    # ... with it, the same bitmap scores the paper's cells that still exist
    r_rows = evaluate_forecasts({"m": fc}, drifted, sample_mask=np.packbits(mask),
                                grid_rows=rows_path, verbose=False)
    r_keys = evaluate_forecasts({"m": fc}, drifted, sample_mask=keys, verbose=False)
    assert r_rows.n_common == r_keys.n_common > 0
    a = r_rows.leaderboard()[lambda d: d.model == "m"]["r2"].iloc[0]
    b = r_keys.leaderboard()[lambda d: d.model == "m"]["r2"].iloc[0]
    assert np.isclose(a, b, equal_nan=True)


def test_grid_rows_is_a_no_op_on_the_canonical_build(tmp_path):
    """Passing the index when the build has NOT drifted must change nothing."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=10, n_q=4, seed=5)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[::2] = True
    rows_path = tmp_path / "rows.parquet"
    bmask.write_grid_rows(grid, rows_path)
    fc = forecast_from_truth(truth, noise=0.2, seed=3)

    plain = evaluate_forecasts({"m": fc}, truth, sample_mask=np.packbits(mask), verbose=False)
    viar = evaluate_forecasts({"m": fc}, truth, sample_mask=np.packbits(mask),
                              grid_rows=rows_path, verbose=False)
    assert plain.n_common == viar.n_common
    assert np.isclose(plain.leaderboard()["r2"].iloc[0],
                      viar.leaderboard()["r2"].iloc[0], equal_nan=True)


def test_grid_rows_error_paths(tmp_path):
    """The guards that keep a bad index from silently misaligning a sample."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=10, n_q=4, seed=5)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[::2] = True
    packed = np.packbits(mask)
    fc = forecast_from_truth(truth, noise=0.2, seed=3)
    rows_path = tmp_path / "rows.parquet"
    bmask.write_grid_rows(grid, rows_path)
    gr = pd.read_parquet(rows_path)

    # missing key columns
    with pytest.raises(ValueError, match="must have"):
        evaluate_forecasts({"m": fc}, truth, sample_mask=packed,
                           grid_rows=gr[["grid_row"]], verbose=False)

    # duplicate (firm, origin) rows in the index
    dupe = pd.concat([gr.iloc[:1], gr.iloc[:-1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_forecasts({"m": fc}, truth, sample_mask=packed,
                           grid_rows=dupe, verbose=False)

    # an index of the right SHAPE but describing entirely different firms
    alien = gr.copy()
    alien["firm"] = "ZZ" + alien["firm"].astype(str)
    with pytest.raises(ValueError, match="no canonical row matched"):
        evaluate_forecasts({"m": fc}, truth, sample_mask=packed,
                           grid_rows=alien, verbose=False)


def test_period_origins_are_rejected_upstream_not_by_the_index(tmp_path):
    """Pins WHY write_grid_rows needs no Period support: canon_origin is
    ``pd.to_datetime(...).dt.to_period("Q")``, which raises on a Period column, so
    TruthGrid rejects such a frame and the dtype can never reach the writer."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = normalize_columns(synthetic_truth(n_firms=6, n_q=4, seed=9))
    as_period = truth.copy()
    as_period["origin"] = pd.PeriodIndex(pd.to_datetime(truth["origin"]), freq="Q")
    with pytest.raises(Exception):
        TruthGrid(as_period)


def test_grid_rows_with_keys_mask_is_reported_not_silent(tmp_path, capsys):
    """Passing --grid-rows with a keys mask does nothing; say so."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=8, n_q=4, seed=5)
    truth_path = tmp_path / "t.parquet"
    truth.to_parquet(truth_path, index=False)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    mask[::2] = True
    keys = bmask.mask_to_keys(mask, grid, truth_path)
    rows_path = tmp_path / "rows.parquet"
    bmask.write_grid_rows(grid, rows_path)
    fc = forecast_from_truth(truth, noise=0.2, seed=3)

    evaluate_forecasts({"m": fc}, truth, sample_mask=keys, grid_rows=rows_path, verbose=True)
    assert "grid-rows ignored" in capsys.readouterr().out


def test_grid_rows_mismatched_release_errors(tmp_path):
    """A row index from a different build cannot align a mask it never described."""
    from proforma20q.truth_grid import TruthGrid
    from proforma20q.schema import normalize_columns

    truth = synthetic_truth(n_firms=10, n_q=4, seed=5)
    grid = TruthGrid(normalize_columns(truth))
    mask = np.zeros(grid.n_cells, dtype=bool)
    other = synthetic_truth(n_firms=7, n_q=4, seed=6)
    rows_path = tmp_path / "rows.parquet"
    bmask.write_grid_rows(TruthGrid(normalize_columns(other)), rows_path)
    fc = forecast_from_truth(truth, noise=0.2, seed=3)

    with pytest.raises(ValueError, match="DIFFERENT releases"):
        evaluate_forecasts({"m": fc}, truth, sample_mask=np.packbits(mask),
                           grid_rows=rows_path, verbose=False)


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
    # every remote-path hint must name an artifact that is actually pinned
    assert set(dl.REMOTE_HINTS) <= set(dl.ARTIFACTS)

    # ...but the placeholder guard itself must still work. Exercise it by putting
    # the module back into the pre-release state, rather than by leaving the repo
    # in it -- otherwise this test would be asserting the release never happened.
    #
    # BOTH network entry points are stubbed to fail loudly rather than left live:
    # the guard firing is what keeps this test offline, and the scenario the test
    # exists for is the guard being BROKEN -- in which case an unstubbed main()
    # would reach the Hub and report an opaque auth error from CI instead of the
    # actual failure. Stubbing keeps it hermetic in both outcomes.
    orig_pins = dict(dl.ARTIFACTS)
    orig_fetch, orig_list = dl.fetch, dl.list_repo_files

    def _boom(*a, **k):
        pytest.fail("placeholder guard did not fire: main() reached the network path")

    dl.ARTIFACTS["full_sample_mask_bits.npy"] = "REPLACE_WITH_MASK_MD5"
    dl.fetch, dl.list_repo_files = _boom, _boom
    try:
        with pytest.raises(SystemExit, match="unfilled release placeholders"):
            dl.main(["--only", "full_sample_mask_bits.npy", "--out", str(tmp_path)])
    finally:
        dl.ARTIFACTS.clear()
        dl.ARTIFACTS.update(orig_pins)
        dl.fetch, dl.list_repo_files = orig_fetch, orig_list


def test_download_artifacts_keeps_the_unverified_bytes_it_points_at(tmp_path):
    """`verify_md5` tells the user the unverified file was "left at <path>", so it
    has to actually be there. The staging tree was wiped on the failure path too,
    which -- after a 7.4 GB download that fails its pin -- pointed the user at a
    file that had just been deleted."""
    dl = _load_script("download_artifacts.py")
    name = "full_sample_mask_bits.npy"
    out = tmp_path / "artifacts"
    out.mkdir()

    def _fake_download(repo_id, remote, repo_type="dataset", local_dir=None):
        p = Path(local_dir) / remote
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"truncated")
        return p

    dl.hf_download = _fake_download          # freshly loaded module; nothing to restore
    with pytest.raises(dl.GatedArtifactError, match="md5 mismatch") as e:
        dl.fetch(name, out, [f"mask/{name}"])

    left = out / f"{name}.unverified"
    assert left.exists() and left.read_bytes() == b"truncated"
    assert str(left) in str(e.value), "the error must name the file that survived"
    assert not (out / name).exists(), "a failed pin must never publish the real name"
    assert not (out / ".hf-part").exists(), "the staging tree is still cleaned up"


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
    hand copy is exactly how the superseded run-1 pin went stale (gh#7; filed
    as gh#15 before the 2026-08 history scrub renumbered the issues).
    Every digest the README quotes must therefore match the
    scripts/download_artifacts.py pin for that file. Nothing is anchored against
    an in-repo copy any more: the Compustat-derived artifacts are gated, so the
    script's pins are the only source of truth.
    Update policy: change scripts/download_artifacts.py first, then make the
    README rows agree -- this test fails on any drift."""
    dl = _load_script("download_artifacts.py")
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"\|\s*\[?`([A-Za-z0-9_./]+)`\]?[^|]*\|[^|]*\|\s*`([0-9a-f]{8})…`", readme)
    listed: dict[str, set[str]] = {}
    for path, pref in rows:
        listed.setdefault(path.rsplit("/", 1)[-1], set()).add(pref)

    anchors: dict[str, set[str]] = {name: {md5} for name, md5 in dl.ARTIFACTS.items()}

    for name, prefs in listed.items():
        assert name in anchors, f"README table digest for {name}, which is not pinned"
        for pref in prefs:
            assert any(full.startswith(pref) for full in anchors[name]), (
                f"README digest `{pref}…` for {name} does not match the "
                f"download_artifacts.py pin")
    for name, md5 in dl.ARTIFACTS.items():
        assert name in listed, f"{name} is pinned in download_artifacts.py but absent from the README tables"
        assert any(md5.startswith(p) for p in listed[name]), \
            f"README never lists the pinned md5 {md5} for {name}"


def test_the_gated_repositories_are_the_only_source_named():
    """The gated Hugging Face repositories are the sole distribution point for
    the Compustat-derived artifacts. An earlier release also mirrored most of
    them to a Zenodo deposit, and the docs offered that as a second place the
    md5 pins would verify against. That is no longer true, and pointing a user
    at an ungated archive for files that are gated for licensing reasons is the
    one stale claim here with legal rather than merely cosmetic consequences --
    so pin its absence across the whole tree, not just the files that happened
    to name it."""
    tracked = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
    offenders = []
    for rel in tracked:
        path = _ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary fixtures and the docs PDF
        if rel == "tests/test_masks.py":
            continue  # this test names it to forbid it
        if re.search(r"zenodo", text, re.IGNORECASE):
            offenders.append(rel)
    assert not offenders, (
        f"{offenders} still name Zenodo; the gated Hugging Face repositories are "
        f"the only source for the Compustat-derived artifacts")

    citation = (_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "huggingface.co/datasets/forma-lab-mccombs/proforma-20q-artifacts" in citation, (
        "CITATION.cff no longer points at the gated dataset repository")


def test_release_documentation_pdf_matches_the_readme_digest():
    """The README block says this repository holds the canonical PDF and that a
    diverging copy elsewhere is the stale one -- but the digest it quotes is
    prose. Hash the file and hold the prose to it, so the claim cannot rot."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    stated = re.findall(r"sha256\s+([0-9a-f]{64})", readme)
    assert stated, "README no longer states a sha256 for release_documentation.pdf"
    pdf = _ROOT / "docs" / "release_documentation.pdf"
    assert pdf.exists(), "docs/release_documentation.pdf is missing from this repository"
    actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert actual in stated, (
        f"docs/release_documentation.pdf hashes to {actual}, which the README does not "
        f"state (it says {stated}). Regenerating the PDF means updating the digest in "
        f"BOTH repositories' READMEs.")
