"""Download + md5-verify the released ProForma-20Q data artifacts from Zenodo.

Nothing WRDS-derived is released. The public artifacts are model OUTPUTS and a
coverage mask (no firm-level Compustat values):

* ``forma_fgrid__pf_full__test__predictions.parquet`` -- the canonical R13 Forma
  5-seed mixture forecast (point track). Pool your model against it to reproduce
  the paper's Panel A / B Full column, or rebuild the pooled mask from it with
  ``scripts/build_full_sample_mask.py``.
* ``full_sample_mask_bits.npy`` -- the 327,244,429-cell Full-sample mask
  (grid-aligned packbits; no firm identifiers). Pass to
  ``proforma20q evaluate --sample-mask``.

Set ``ZENODO_RECORD`` below to the published Zenodo record id (see the DOI on the
repo's release / README), then::

    python scripts/download_artifacts.py --out data/artifacts
    python scripts/download_artifacts.py --only full_sample_mask_bits.npy
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

# The Zenodo record id of the published artifact bundle (the number in the DOI
# 10.5281/zenodo.<ZENODO_RECORD>). Filled in at release time.
ZENODO_RECORD = "REPLACE_WITH_ZENODO_RECORD_ID"

# filename -> md5 (pinned; the mask hash also lives in full_sample_mask.manifest.json)
ARTIFACTS: dict[str, str] = {
    "forma_fgrid__pf_full__test__predictions.parquet": "c4f0f721409bdfe32c215ddc72c430da",
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
    print(f"downloading {name} from {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)  # streams to disk
    print()
    got = md5sum(dest)
    if got != expected:
        raise SystemExit(f"md5 mismatch for {name}: got {got}, expected {expected}")
    print(f"[ok] {name}: verified ({expected})")
    return dest


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
    for name in (args.only or list(ARTIFACTS)):
        fetch(name, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
