"""Build the dollar-space ground-truth panel that the Forma USD forecasts merge against.

The released dollar forecasts
([`forma-lab-mccombs/forma-usd-forecasts`](https://huggingface.co/datasets/forma-lab-mccombs/forma-usd-forecasts))
carry predictions in millions of USD but no realized values -- those are
verbatim Compustat, which we cannot redistribute. This rebuilds them from YOUR
OWN licensed pull, keyed identically, so the merge is a one-liner.

WHY THIS EXISTS: 26 of the 78 pf_full targets either do not exist in Compustat
under those names, or do not mean the same thing under them, and merging the
obvious column is silently wrong.

  * 20 are QUARTERLY FLOWS de-cumulated from a fiscal-year-to-date item:
    ``oancfq`` <- ``oancfy``, ``capxq`` <- ``capxy``, and so on. Compustat ships
    only the YTD version. Joining ``oancfy`` onto a forecast of ``oancfq``
    compares a quarterly flow against a year-to-date actual -- both numeric,
    both $M, nothing errors, and the error grows through the fiscal year.
  * 6 are COMPUTED items: ``gpq``, ``fcfq``, ``wcapq``, ``aoq_ex_intanq``,
    ``loq_ex_dr``, ``xsgaq_ex_rd``.
  * ``wcapq`` is the trap. Compustat ships a column by that name, so it merges
    cleanly and never errors -- but it means something else. The benchmark
    DEFINES it as ``actq - lctq``, and that is the series the forecasts are of.
    ``add_computed_features`` overwrites the native column for exactly this
    reason; see ``test_computed_features_overwrite_native_columns``.

Nothing here restates those definitions. The de-cumulation, the formula table
and the target universe are imported from ``proforma20q``, which is the same
code path ``proforma20q build`` uses, so this script cannot drift from the
benchmark it is producing truth for.

KEYS. ``quarter`` in the release is ``datadate`` snapped to CALENDAR quarter end
(``datadate.dt.to_period('Q').dt.end_time``). Merge the forecast's
``target_quarter`` against this file's ``quarter`` -- NOT against ``quarter``,
which in the forecast file is the origin.

Usage
-----
  python scripts/build_usd_truth.py \
      --compustat /path/to/your/compustat_fundq.parquet \
      --out data/usd_truth.parquet

Then, for one account:

  import pandas as pd
  fc = pd.read_parquet("hf://datasets/forma-lab-mccombs/forma-usd-forecasts/target=revtq/")
  tr = pd.read_parquet("data/usd_truth.parquet")
  tr = tr[tr.target == "revtq"]
  m  = fc.merge(tr, left_on=["firm_id", "target_quarter"],
                    right_on=["firm_id", "quarter"], suffixes=("", "_t"))
  err = m.pred_p50_musd - m.actual_musd

``target`` is the dataset's Hive partition key, so a single-partition read has no
such column and it does not belong in the join keys. A filtered read from the
dataset root restores it -- see the dataset card.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proforma20q.build import (  # noqa: E402
    add_computed_features,
    convert_ytd_to_quarterly,
)
from proforma20q.config import pf_full_targets  # noqa: E402


def build_usd_truth(raw: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Raw fundq pull -> long ``(firm_id, quarter, target, actual_musd)``.

    Mirrors the build's own preparation: de-cumulate YTD sources, then recompute
    every computed item (overwriting any like-named native column), then melt.
    """
    need = {"gvkey", "datadate"}
    missing = need - set(raw.columns)
    if missing:
        raise SystemExit(f"missing required column(s): {sorted(missing)}")
    for c in ("fyearq", "fqtr"):
        if c not in raw.columns:
            raise SystemExit(
                f"missing '{c}'. The YTD de-cumulation is fiscal-year aware and "
                f"cannot be done without it; re-pull with fyearq/fqtr.")

    df = raw.rename(columns={"gvkey": "firm_id", "datadate": "quarter"}).copy()
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.end_time

    n0 = len(df)
    df = convert_ytd_to_quarterly(df, firm_col="firm_id")
    if len(df) != n0:
        print(f"  dropped {n0 - len(df):,} duplicate (firm, calendar quarter) rows")

    universe = pf_full_targets()
    df = add_computed_features(df, universe)

    wanted = list(targets) if targets else universe
    have = [t for t in wanted if t in df.columns]
    absent = sorted(set(wanted) - set(have))
    if targets and absent:
        raise SystemExit(f"requested targets not derivable from this pull: {absent}")
    if absent:
        print(f"  {len(absent)} target(s) not derivable from this pull: {absent}")

    fid = df["firm_id"].astype(str).str.strip().str.zfill(6)
    long = (df[have]
            .assign(firm_id=fid.to_numpy(), quarter=df["quarter"].to_numpy())
            .melt(id_vars=["firm_id", "quarter"], var_name="target",
                  value_name="actual_musd")
            .dropna(subset=["actual_musd"]))
    long["actual_musd"] = long["actual_musd"].astype(np.float32)
    long = long.sort_values(["firm_id", "quarter", "target"], ignore_index=True)

    dup = long.duplicated(["firm_id", "quarter", "target"]).sum()
    if dup:
        raise SystemExit(f"{dup:,} duplicate (firm_id, quarter, target) rows -- "
                         f"the de-duplication did not resolve this pull")
    return long


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compustat", required=True,
                    help="your licensed Compustat Fundamentals Quarterly pull "
                         "(parquet or csv); needs gvkey, datadate, fyearq, fqtr "
                         "and the item columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default=None,
                    help="comma list to restrict output (default: all 78)")
    args = ap.parse_args()

    src = Path(args.compustat)
    print(f"Reading {src}", flush=True)
    if src.suffix.lower() in (".csv", ".txt", ".gz"):
        raw = pd.read_csv(src, low_memory=False)
    else:
        raw = pd.read_parquet(src)
    print(f"  {len(raw):,} rows x {len(raw.columns)} columns")

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    long = build_usd_truth(raw, targets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, compression="zstd", index=False)
    print(f"\nWrote {len(long):,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    print(f"  firms {long['firm_id'].nunique():,}  "
          f"quarters {long['quarter'].nunique()}  "
          f"targets {long['target'].nunique()}")
    print("\nMerge with:  forecast.target_quarter  ==  truth.quarter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
