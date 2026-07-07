"""(Maintainer) Build the paper's Full-sample pooled mask.

The ICAIF'26 headline table scores each panel on an exact pooled *common sample*:
a cell counts only if every pooled model forecasts it finitely. For the **Full**
column that pool is 8 models, but their coverage has a single binding constraint
--- the reference model ``forma_fgrid`` --- and every other full-coverage model
is a superset of it. So the Full sample is exactly::

    forma_fgrid finite-prediction cells  INTERSECT  truth-finite cells
    (truth-finite == the target actual AND the origin baseline both present)

which this script materialises and verifies against the published count
(327,244,429). It reads the forecast file read-only and streams it in batches.

The mask ships in two interchangeable forms (see ``proforma20q evaluate
--sample-mask``):

* a **grid-aligned bit array** (``mask_bits.npy``, ``np.packbits`` of the
  canonical cell order) --- compact (~66 MB), contains no firm identifiers, but
  must be applied against the canonical ``tabular_test``;
* a **keys** table (``mask_keys.parquet`` with ``firm,target,origin,horizon``)
  --- portable / matches by value, larger.

Usage::

    python scripts/build_full_sample_mask.py \
        --forecast <...>/forma_fgrid__pf_full__test__predictions.parquet \
        --truth    data/processed/tabular_test__pf_full__r13_node_optionD_indfe_val8.parquet \
        --out      results/masks --keys --expect 327244429
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from proforma20q.schema import PARQUET_ENGINE, normalize_columns
from proforma20q.truth_grid import TruthGrid


def build_mask(forecast_path, truth_path, *, batch_size: int = 8_000_000, verbose: bool = True):
    """Return ``(mask_bool, grid)`` for ``forecast finite INTERSECT truth finite``."""
    log = print if verbose else (lambda *a, **k: None)
    truth = pd.read_parquet(truth_path, engine=PARQUET_ENGINE)
    grid = TruthGrid(normalize_columns(truth))

    truth_finite = np.zeros(grid.n_cells, dtype=bool)
    for _t, _h, sl, act, base in grid.iter_blocks():
        truth_finite[sl] = np.isfinite(act) & np.isfinite(base)

    covered = np.zeros(grid.n_cells, dtype=bool)
    pf = pq.ParquetFile(forecast_path)
    names = set(pf.schema_arrow.names)

    def pick(*opts):
        return next(c for c in opts if c in names)

    # accept the internal Forma schema (firm_id/quarter/forecast_horizon) or the
    # submission schema (firm/origin/horizon); normalize_columns maps either.
    cols = [pick("firm_id", "firm"), "target", pick("quarter", "origin"),
            pick("forecast_horizon", "horizon"), "prediction"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = normalize_columns(batch.to_pandas())
        pred = pd.to_numeric(df["prediction"], errors="coerce").to_numpy(np.float64)
        row_id, valid = grid.map_forecast_rows(df)
        ok = valid & np.isfinite(pred)
        # OR over all finite rows for a key. The scoring path (build_residual) keeps
        # the FIRST occurrence of a duplicate key, so a mask cell could in principle
        # be more generous than scoring if a key's first row were NaN and a later
        # one finite; evaluate then intersects the mask with the models' common
        # sample, so an over-generous cell is dropped, never wrongly scored. The
        # `--expect` count assertion is the guard that no such duplicate skews it.
        covered[row_id[ok]] = True

    mask = covered & truth_finite
    log(f"forecast coverage {int(covered.sum()):,} INTERSECT truth-finite "
        f"{int(truth_finite.sum()):,} -> mask {int(mask.sum()):,} cells")
    return mask, grid


def mask_to_keys(mask: np.ndarray, grid: TruthGrid, truth_path) -> pd.DataFrame:
    truth = normalize_columns(pd.read_parquet(truth_path, engine=PARQUET_ENGINE))
    cells = np.flatnonzero(mask)
    block, wide_row = np.divmod(cells, grid.n_wide)
    t_id, h_idx = np.divmod(block, grid.n_h)
    firm = truth["firm"].to_numpy()[wide_row]
    origin = pd.to_datetime(truth["origin"]).to_numpy()[wide_row]
    return pd.DataFrame({
        "firm": firm,
        "target": pd.Categorical.from_codes(t_id.astype(np.int32), categories=grid.targets),
        "origin": origin,
        "horizon": np.asarray(grid.horizons)[h_idx].astype(np.int16),
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", required=True, help="binding model forecast (forma_fgrid ensemble)")
    ap.add_argument("--truth", required=True, help="tabular_test artifact (eval ground truth)")
    ap.add_argument("--out", default="results/masks")
    ap.add_argument("--keys", action="store_true", help="also write the portable keys parquet")
    ap.add_argument("--expect", type=int, default=None, help="assert the cell count (e.g. 327244429)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mask, grid = build_mask(args.forecast, args.truth)
    n = int(mask.sum())
    if args.expect is not None and n != args.expect:
        raise SystemExit(f"mask has {n:,} cells, expected {args.expect:,}")

    bits_path = out / "full_sample_mask_bits.npy"
    np.save(bits_path, np.packbits(mask))
    md5 = hashlib.md5(Path(bits_path).read_bytes()).hexdigest()
    print(f"wrote {bits_path} ({n:,} cells)  md5={md5}")
    if args.keys:
        keys_path = out / "full_sample_mask_keys.parquet"
        mask_to_keys(mask, grid, args.truth).to_parquet(keys_path, index=False)
        print(f"wrote {keys_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
