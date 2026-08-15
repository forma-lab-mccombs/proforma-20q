"""Download + md5-verify the released ProForma-20Q data artifacts.

The artifacts live in the gated Hugging Face dataset repository
``forma-lab-mccombs/proforma-20q-artifacts``. The gate is ``auto``: any Hugging
Face account can accept the terms and get immediate access. They are gated
rather than bundled because they are derived from Compustat via WRDS, whose
licence is non-commercial and therefore incompatible with the Apache-2.0 grant
covering this repository's code. See README.md and NOTICE.

Every file is verified against the md5 pinned below, so a truncated or
substituted download fails loudly rather than scoring silently against the
wrong bytes. The pins are unchanged from the Zenodo deposit this bundle mirrors
(10.5281/zenodo.21269003) -- the published copies are byte-identical to the
archival record, so a file fetched either way verifies against the same digest.

The bundle is one file per point-track exhibit of Table 1, plus the
density-family sidecar, the coverage mask, and its row index:

* ``forma_fgrid__pf_full__test__predictions.parquet`` -- the canonical R13
  Forma 5-seed Gaussian mixture forecast (squared-error track). Pool your
  model against it to reproduce the paper's Panel A Full column, or rebuild
  the pooled mask from it with ``scripts/build_full_sample_mask.py``.
* ``ffnn_linear_b50__pf_full__test__predictions.parquet`` and
  ``ffnn_large_b50__pf_full__test__predictions.parquet`` -- the two FFNN
  5-seed mixture forecasts (the Panel A comparator rows).
* ``forma_lap05_fgrid__pf_full__test__predictions.parquet`` -- the canonical
  R13 Forma Laplace mixture (absolute-error track; the Panel B Full column),
  plus its ``forma_lap05_fgrid__pf_full__test__predictions.nll.json`` family
  sidecar. KEEP THE SIDECAR NEXT TO THE PARQUET: the evaluator resolves the
  density family from ``{stem}.nll.json`` and silently defaults to Gaussian
  when it is missing.
* ``full_sample_mask_bits.npy`` -- the 327,244,429-cell Full-sample mask
  (grid-aligned packbits; no firm identifiers). Pass to
  ``proforma20q evaluate --sample-mask``.
* ``full_sample_grid_rows.parquet`` -- the canonical row index (``grid_row,
  firm, origin``) the mask is a bitmap over. Pass alongside the mask as
  ``evaluate --grid-rows`` and the bitmap applies to a vintage-drifted rebuild,
  matched by ``(firm, origin)`` value. Only 1.9 MB, and it is what keeps
  reproducing the paper's sample from costing a 3.7 GB forecast download; it
  cannot be derived from the forecast, which omits 23,970 canonical rows it
  never forecast. It carries gvkeys, hence the gate.

The regularization statistics are NOT here. They ship from the gated model
repository ``forma-lab-mccombs/forma`` as ``reference/regularization_stats.parquet``,
co-located with the weights so the ``(mu, sigma, k)`` defining the target space
have a single source of truth; ``proforma20q build --reg-stats canonical``
fetches them for you.

The density track (Panel C -- exact mixture NLL/CRPS over the per-seed
forecasts) is out of scope for this release; issue #3 tracks it.

Usage::

    python scripts/download_artifacts.py --out data/artifacts
    python scripts/download_artifacts.py --only full_sample_mask_bits.npy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proforma20q.gated import (  # noqa: E402
    ARTIFACTS_REPO, GatedArtifactError, hf_download, list_repo_files,
    md5sum, repo_url, resolve_remote_path, verify_md5,
)

# filename -> md5 (pinned; the mask hash also lives in full_sample_mask.manifest.json).
# Digests verified 2026-07-27 directly over the canonical store, whose
# MANIFEST.tsv records the same values, and unchanged by the move to Hugging
# Face: the published copies mirror the archival deposit byte-for-byte.
ARTIFACTS: dict[str, str] = {
    "forma_fgrid__pf_full__test__predictions.parquet": "1820fcc90e71989af558f9d103d6fc31",
    "ffnn_linear_b50__pf_full__test__predictions.parquet": "e419c8330ff6c9c6396a7d2e04f05c3e",
    "ffnn_large_b50__pf_full__test__predictions.parquet": "915779a3ff79b6e344d45910ac5e4026",
    "forma_lap05_fgrid__pf_full__test__predictions.parquet": "1e8b0415905eeac7cf46b052f5c1cbf5",
    "forma_lap05_fgrid__pf_full__test__predictions.nll.json": "a3d8659a201a2081dd693a8f0de051c3",
    "full_sample_mask_bits.npy": "a36008d8dbfeb56992f1049fd543d781",
    "full_sample_grid_rows.parquet": "adbc2ae6eef7f23b3576af525cfbeeec",
}

# Where each artifact is expected to sit inside the repository. Only a hint:
# anything not found at this exact path is resolved by unique basename against
# the repo listing, so publishing under a different prefix does not break this
# script. Files land in --out FLAT regardless, because that is what
# `evaluate --sample-mask` and friends are documented to take.
REMOTE_HINTS: dict[str, str] = {
    "full_sample_mask_bits.npy": "mask/full_sample_mask_bits.npy",
    "full_sample_grid_rows.parquet": "mask/full_sample_grid_rows.parquet",
}


def fetch(name: str, out_dir: Path, listing: list[str]) -> Path:
    expected = ARTIFACTS[name]
    dest = out_dir / name
    if dest.exists() and md5sum(dest) == expected:
        print(f"[ok] {name}: already present and verified")
        return dest

    remote = resolve_remote_path(REMOTE_HINTS.get(name, name), listing,
                                 ARTIFACTS_REPO, "dataset")
    # Stage inside the output directory (same filesystem, so publishing is an
    # atomic rename rather than a second full-size copy) and move to the real
    # filename only after the md5 passes, so an interrupted or corrupt transfer
    # can never leave a plausible-looking artifact where a later run trusts it.
    staging = out_dir / ".hf-part"
    print(f"downloading {name} from {ARTIFACTS_REPO}:{remote}")
    got = hf_download(ARTIFACTS_REPO, remote, repo_type="dataset", local_dir=staging)
    try:
        # Move out of the staging tree BEFORE verifying, so the file `verify_md5`
        # reports as "left at <path>" survives the cleanup below. A 7.4 GB
        # download that fails its pin is worth inspecting; deleting it and then
        # pointing the user at the path would be the worst of both.
        unverified = out_dir / f"{name}.unverified"
        os.replace(got, unverified)
        verify_md5(unverified, expected, label=name)
        os.replace(unverified, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[ok] {name}: verified ({expected})")
    return dest


def _with_sidecars(names) -> list[str]:
    """Append each selected parquet's ``{stem}.nll.json`` family sidecar.

    The evaluator resolves the density family from the sidecar and silently
    defaults to Gaussian when it is missing, so ``--only <panel B parquet>``
    must not be able to fetch the forecast without it.
    """
    selected = list(names)
    for name in list(selected):
        side = name.removesuffix(".parquet") + ".nll.json"
        if name.endswith(".parquet") and side in ARTIFACTS and side not in selected:
            selected.append(side)
    return selected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/artifacts", help="download directory")
    ap.add_argument("--only", nargs="*", choices=list(ARTIFACTS), help="subset to fetch")
    args = ap.parse_args(argv)

    unset = [k for k, v in ARTIFACTS.items() if str(v).startswith("REPLACE_WITH")]
    if unset:
        raise SystemExit("This script has unfilled release placeholders: " + ", ".join(unset)
                         + ". Set the md5s (see README) first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = _with_sidecars(args.only or ARTIFACTS)

    # One listing up front: it resolves every remote path, and it is also where
    # a missing token or an unaccepted gate surfaces -- once, with instructions,
    # instead of once per file mid-download.
    try:
        listing = list_repo_files(ARTIFACTS_REPO, repo_type="dataset")
        for name in wanted:
            fetch(name, out, listing)
    except GatedArtifactError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    print(f"\n{len(wanted)} artifact(s) verified in {out}/  "
          f"(source: {repo_url(ARTIFACTS_REPO, 'dataset')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
