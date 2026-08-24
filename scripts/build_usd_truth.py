"""Build the dollar-space ground-truth panel that the Forma USD forecasts merge against.

The released dollar forecasts
([`forma-lab-mccombs/forma-usd-forecasts`](https://huggingface.co/datasets/forma-lab-mccombs/forma-usd-forecasts))
carry predictions in millions of USD but no realized values -- those are
verbatim Compustat, which cannot be redistributed. This rebuilds them from YOUR
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

Nothing here restates those definitions. The de-cumulation, the formula table,
the format filters and the target universe are imported from ``proforma20q``,
which is the same code path ``proforma20q build`` uses, so this script cannot
drift from the benchmark it is producing truth for.

A COMPUTED TARGET IS EMITTED ONLY IF IT WAS ACTUALLY COMPUTED. ``build``'s
``add_computed_features`` recomputes an item only when every input column is
present and is silent otherwise, which on a thin pull would leave Compustat's
native ``wcapq`` sitting in the frame to be emitted as "truth" -- the precise
failure this script exists to prevent. So every computed target is checked
against its own requires-list: unresolvable ones are dropped and named, and
naming one explicitly in ``--targets`` is an error rather than a silent
substitution.

YOUR PULL MUST BE FILTERED THE WAY THE BENCHMARK'S IS. ``comp.fundq`` carries
several format variants of the same firm-quarter; the canonical universe is
``indfmt=INDL, datafmt=STD, consol=C, popsrc=D`` (``task.yaml``,
``universe.compustat_filters``). If those columns are in your pull this script
applies the filters itself. If they are absent it cannot, and says so --
de-duplication would then keep whichever variant happened to sort last, which is
an artefact of your query's row order, not a property of the data.

Rows outside the benchmark's own sample (the SIC 6000-6999 financial exclusion,
the CRSP link windows) are NOT removed. They are harmless -- the forecasts do
not cover those firm-quarters, so the merge drops them -- and removing them
would need CRSP data this script does not ask for.

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

Exit codes follow AGENTS.md section 5: ``0`` success, ``2`` precondition missing
(columns absent, nothing derivable, empty result).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proforma20q.build import (  # noqa: E402
    _computed_definitions,
    add_computed_features,
    convert_ytd_to_quarterly,
)
from proforma20q.config import feature_set_items, load_task_config  # noqa: E402

# _computed_definitions is private, and imported anyway on purpose: the
# alternative is restating each computed item's requires-list here, which is the
# one kind of drift this script is built to be incapable of.


class Precondition(SystemExit):
    """Exit 2 -- 'precondition missing' per AGENTS.md section 5."""

    def __init__(self, msg: str):
        print(f"error: {msg}", file=sys.stderr)
        super().__init__(2)


def benchmark_targets() -> list[str]:
    """The task config's own feature set, not a hardcoded 'pf_full'."""
    return feature_set_items(load_task_config()["benchmark"]["feature_set"])


def apply_compustat_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the canonical format variant, when the pull carries the columns."""
    filters = load_task_config()["universe"]["compustat_filters"]
    present = [c for c in filters if c in df.columns]
    if not present:
        print("  WARNING: none of "
              f"{sorted(filters)} present -- cannot restrict to the canonical "
              "format variant. If your pull has more than one variant per "
              "firm-quarter, which one becomes 'truth' depends on your query's "
              "row order. Re-pull with those columns, or filter them yourself.")
        return df
    if len(present) < len(filters):
        print(f"  WARNING: only {present} present of {sorted(filters)}; "
              "filtering on what is available")
    keep = np.ones(len(df), dtype=bool)
    for c in present:
        keep &= df[c].astype(str).str.strip().eq(str(filters[c])).to_numpy()
    n = int((~keep).sum())
    if n:
        print(f"  dropped {n:,} rows outside "
              + ", ".join(f"{c}={filters[c]}" for c in present))
    return df.loc[keep]


def normalize_gvkey(s: pd.Series) -> pd.Series:
    """gvkey -> zero-padded 6-character string, as the release keys it.

    A CSV pull with any missing gvkey types the column float64, so a plain
    ``astype(str)`` yields ``'1690.0'`` -- already 6 characters, so ``zfill`` is
    a no-op and the id merges against nothing.
    """
    # float64 (any missing gvkey in a CSV types the whole column that way)
    # stringifies as '1690.0'; the trailing-zero strip is what handles it, and
    # anything the strip cannot rescue -- 'nan', '1.7e+05' -- fails the digit
    # check below rather than silently becoming an id that merges against nothing.
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    bad = ~out.str.fullmatch(r"\d{1,6}")
    if bad.any():
        raise Precondition(
            f"{int(bad.sum()):,} gvkey value(s) are not 1-6 digits, e.g. "
            f"{out[bad].unique()[:5].tolist()}")
    return out.str.zfill(6)


def resolve_targets(df: pd.DataFrame, wanted: list[str], *,
                    explicit: bool) -> list[str]:
    """Targets we can honestly emit. A computed item whose inputs are absent is
    NOT emitted, even when a like-named native column would satisfy the lookup."""
    computed = _computed_definitions()
    unresolvable, absent = {}, []
    for t in wanted:
        if t in computed:
            missing = [c for c in computed[t][0] if c not in df.columns]
            if missing:
                unresolvable[t] = missing
                continue
        if t not in df.columns:
            absent.append(t)
    if unresolvable and explicit:
        raise Precondition(
            "computed target(s) cannot be recomputed from this pull, and a "
            "native Compustat column of the same name is NOT the same series: "
            + "; ".join(f"{t} needs {m}" for t, m in sorted(unresolvable.items())))
    if absent and explicit:
        raise Precondition(f"requested targets not in this pull: {sorted(absent)}")
    for t, missing in sorted(unresolvable.items()):
        native = " (a native column of that name is present and is NOT it)" \
            if t in df.columns else ""
        print(f"  SKIP {t}: needs {missing}{native}")
    if absent:
        print(f"  {len(absent)} target(s) not in this pull: {sorted(absent)}")
    keep = [t for t in wanted if t not in unresolvable and t not in absent]
    return keep


def build_usd_truth(raw: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Raw fundq pull -> long ``(firm_id, quarter, target, actual_musd)``."""
    for c in ("gvkey", "datadate"):
        if c not in raw.columns:
            raise Precondition(f"missing required column '{c}'")
    for c in ("fyearq", "fqtr"):
        if c not in raw.columns:
            raise Precondition(
                f"missing '{c}'. The YTD de-cumulation is fiscal-year aware and "
                f"cannot be done without it; re-pull with fyearq/fqtr.")

    df = raw.rename(columns={"gvkey": "firm_id", "datadate": "quarter"}).copy()
    n_null = int(df["firm_id"].isna().sum())
    if n_null:
        print(f"  dropped {n_null:,} rows with no gvkey")
        df = df[df["firm_id"].notna()]
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.end_time

    df = apply_compustat_filters(df)
    if df.empty:
        raise Precondition("no rows left after the canonical format filters")

    n0 = len(df)
    # de-cumulate BEFORE computing: fcfq = oancfq - capxq needs quarterly inputs
    df = convert_ytd_to_quarterly(df, firm_col="firm_id")
    if len(df) != n0:
        print(f"  de-duplicated {n0 - len(df):,} (firm, calendar quarter) rows")

    universe = benchmark_targets()
    df = add_computed_features(df, universe)

    have = resolve_targets(df, list(targets) if targets else universe,
                           explicit=targets is not None)
    if not have:
        raise Precondition("no targets derivable from this pull")

    fid = normalize_gvkey(df["firm_id"]).to_numpy()
    qtr = df["quarter"].to_numpy()

    # Melt one target at a time. Melting all 78 at once materializes
    # n_rows x 78 before the dropna -- ~133 M rows on the canonical panel.
    frames = []
    for t in have:
        vals = df[t].to_numpy(dtype="float64", na_value=np.nan, copy=False)
        keep = np.isfinite(vals)
        if not keep.any():
            continue
        frames.append(pd.DataFrame({
            "firm_id": fid[keep], "quarter": qtr[keep], "target": t,
            "actual_musd": vals[keep].astype(np.float32)}))
    if not frames:
        raise Precondition("every derivable target is entirely missing")

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["firm_id", "quarter", "target"], ignore_index=True)

    # Invariant, not input validation: convert_ytd_to_quarterly has already
    # made (firm_id, quarter) unique, and each frame above is one target, so a
    # duplicate here would mean that de-duplication changed behaviour upstream.
    dup = int(long.duplicated(["firm_id", "quarter", "target"]).sum())
    if dup:
        raise AssertionError(
            f"{dup:,} duplicate (firm_id, quarter, target) rows after melt -- "
            f"the (firm, quarter) de-duplication upstream no longer holds")
    return long


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compustat", required=True,
                    help="your licensed Compustat Fundamentals Quarterly pull "
                         "(parquet or csv); needs gvkey, datadate, fyearq, fqtr, "
                         "the item columns, and ideally indfmt/datafmt/consol/"
                         "popsrc so the canonical variant can be selected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default=None,
                    help="comma list to restrict output (default: all 78)")
    ap.add_argument("--engine", default="fastparquet",
                    help="parquet engine (default: fastparquet, as the package uses)")
    args = ap.parse_args()

    src = Path(args.compustat)
    if not src.is_file():
        raise Precondition(f"no such file: {src}")
    print(f"Reading {src}", flush=True)
    if src.suffix.lower() in (".csv", ".txt", ".gz"):
        raw = pd.read_csv(src, low_memory=False)
    else:
        raw = pd.read_parquet(src, engine=args.engine)
    print(f"  {len(raw):,} rows x {len(raw.columns)} columns")

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    long = build_usd_truth(raw, targets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, engine=args.engine, compression="zstd", index=False)
    print(f"\nWrote {len(long):,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    print(f"  firms {long['firm_id'].nunique():,}  "
          f"quarters {long['quarter'].nunique()}  "
          f"targets {long['target'].nunique()}")
    print("\nMerge with:  forecast.target_quarter  ==  truth.quarter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
