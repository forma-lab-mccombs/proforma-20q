"""Download + md5-verify the released ProForma-20Q data artifacts from Zenodo.

    ============================ NOT YET ACTIVE ============================
    PLACEHOLDER. The artifacts are deposited to an archival record (Zenodo)
    at paper submission. Until then no record exists, ``ZENODO_RECORD``
    below is an UNFILLED placeholder, and this script intentionally refuses to
    run -- there is nothing to download yet. The md5s ARE final, so this script
    becomes live the moment the record id is filled in.
    =======================================================================

Nothing WRDS-derived is released. The public artifacts are model OUTPUTS and a
coverage mask (no firm-level Compustat values) -- one file per point-track
exhibit of Table 1, plus the density-family sidecar and the mask:

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

The density track (Panel C -- exact mixture NLL/CRPS over the per-seed
forecasts) is out of scope for this release; issue #3 tracks it.

Set ``ZENODO_RECORD`` below to the published Zenodo record id (see the DOI on the
repo's release / README), then::

    python scripts/download_artifacts.py --out data/artifacts
    python scripts/download_artifacts.py --only full_sample_mask_bits.npy
"""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

# The Zenodo record id of the published artifact bundle (the number in the DOI
# 10.5281/zenodo.<ZENODO_RECORD>). Filled in at deposit time (paper submission).
ZENODO_RECORD = "REPLACE_WITH_ZENODO_RECORD_ID"

# filename -> md5 (pinned; the mask hash also lives in full_sample_mask.manifest.json).
# Digests verified 2026-07-27 directly over the canonical store, whose
# MANIFEST.tsv records the same values.
ARTIFACTS: dict[str, str] = {
    "forma_fgrid__pf_full__test__predictions.parquet": "1820fcc90e71989af558f9d103d6fc31",
    "ffnn_linear_b50__pf_full__test__predictions.parquet": "e419c8330ff6c9c6396a7d2e04f05c3e",
    "ffnn_large_b50__pf_full__test__predictions.parquet": "915779a3ff79b6e344d45910ac5e4026",
    "forma_lap05_fgrid__pf_full__test__predictions.parquet": "1e8b0415905eeac7cf46b052f5c1cbf5",
    "forma_lap05_fgrid__pf_full__test__predictions.nll.json": "a3d8659a201a2081dd693a8f0de051c3",
    "full_sample_mask_bits.npy": "a36008d8dbfeb56992f1049fd543d781",
}


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _progress(count, block_size, total):
    if total > 0:
        pct = min(100, 100 * count * block_size / total)
        print(f"\r    {pct:5.1f}%", end="", flush=True)


def fetch(name: str, out_dir: Path) -> Path:
    expected = ARTIFACTS[name]
    dest = out_dir / name
    if dest.exists() and md5sum(dest) == expected:
        print(f"[ok] {name}: already present and verified")
        return dest
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
    # Stage under a .part name and rename only after the md5 passes, so an
    # interrupted or corrupt transfer can never leave a plausible-looking
    # artifact at the real filename.
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {name} from {url}")
    urllib.request.urlretrieve(url, part, reporthook=_progress)  # streams to disk
    print()
    got = md5sum(part)
    if got != expected:
        raise SystemExit(f"md5 mismatch for {name}: got {got}, expected {expected} "
                         f"(unverified download left at {part})")
    os.replace(part, dest)
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

    unset = [k for k, v in [("ZENODO_RECORD", ZENODO_RECORD)] + list(ARTIFACTS.items())
             if str(v).startswith("REPLACE_WITH")]
    if unset:
        raise SystemExit("This script has unfilled release placeholders: " + ", ".join(unset)
                         + ". Set the Zenodo record id and md5s (see README) first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in _with_sidecars(args.only or ARTIFACTS):
        fetch(name, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
