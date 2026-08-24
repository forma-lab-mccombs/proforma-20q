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
    ``oancfq`` <- ``oancfy``, ``capxq`` <- ``capxy``, and so on. Joining
    ``oancfy`` onto a forecast of ``oancfq`` compares a quarterly flow against a
    year-to-date actual -- both numeric, both $M, nothing errors, and the error
    grows through the fiscal year.
  * 6 are COMPUTED items: ``gpq``, ``fcfq``, ``wcapq``, ``aoq_ex_intanq``,
    ``loq_ex_dr``, ``xsgaq_ex_rd``.
  * ``wcapq`` is the trap. Compustat ships a column by that name, so it merges
    cleanly and never errors -- but it means something else. The benchmark
    DEFINES it as ``actq - lctq``, and that is the series the forecasts are of.

Nothing here restates those definitions. The de-cumulation, the formula table,
the YTD base list, the format filters and the target universe are imported from
``proforma20q``, which is the same code path ``proforma20q build`` uses, so this
script cannot drift from the benchmark it is producing truth for.

A TARGET IS EMITTED ONLY IF IT WAS ACTUALLY DERIVED. Both of the build's
preparation steps are silent no-ops when their inputs are absent:
``add_computed_features`` skips an item whose formula inputs are missing, and
``convert_ytd_to_quarterly`` skips a base whose ``{base}y`` source is missing. On
a pull that carries a native ``wcapq`` or a native ``oancfq``, either silence
would leave a differently-defined column sitting in the frame to be emitted as
"truth" -- the precise failure this script exists to prevent. So every one of the
26 is checked against what it actually needed: unresolvable ones are dropped and
named, and naming one in ``--targets`` is an error rather than a substitution.

That check is over the TRANSITIVE input closure, not the target alone.
``fcfq = oancfq - capxq`` is why: both of its inputs are themselves YTD-derived,
so on a pull with native ``oancfq``/``capxq`` and no ``oancfy``/``capxy`` a
one-level requires-check passes while the formula quietly consumes two columns
the de-cumulation never produced -- and the script would refuse to give you
``oancfq`` while handing you a difference of it.

YOUR PULL MUST BE FILTERED THE WAY THE BENCHMARK'S IS. ``comp.fundq`` carries
several format variants of the same firm-quarter; the canonical universe is
``indfmt=INDL, datafmt=STD, consol=C, popsrc=D`` (``task.yaml``,
``universe.compustat_filters``). If those columns are in your pull this script
applies the filters itself. If they are absent it cannot, and says so --
de-duplication would then keep whichever variant happened to sort last, which is
an artefact of your query's row order, not a property of the data.

Rows outside the benchmark's own sample (the SIC 6000-6999 financial exclusion,
the CRSP link windows) are NOT removed. They are harmless -- the merge is an
inner join, so firm-quarters the forecasts do not cover simply do not match --
and removing them would need CRSP data this script does not ask for. The closing
firm/quarter counts therefore describe this file, not the mergeable panel.

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
(columns absent, nothing derivable, a requested target not obtainable).
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
    _ytd_bases,
    add_computed_features,
    convert_ytd_to_quarterly,
)
from proforma20q.config import feature_set_items, load_task_config  # noqa: E402

# _computed_definitions and _ytd_bases are private, and imported anyway on
# purpose: the alternative is restating each computed item's requires-list and
# the YTD base list here, which is the one kind of drift this script is built to
# be incapable of.


def say(msg: str) -> None:
    """Diagnostics, flushed. Unflushed stdout through a pipe lands *after* the
    unbuffered stderr an error writes -- AGENTS.md section 4."""
    print(msg, flush=True)


class Precondition(SystemExit):
    """Exit 2 -- 'precondition missing' per AGENTS.md section 5."""

    def __init__(self, msg: str):
        sys.stdout.flush()
        print(f"error: {msg}", file=sys.stderr, flush=True)
        super().__init__(2)


#: Returned by ``why_unresolvable`` for a target that is simply not in the pull,
#: as opposed to one that is present but cannot honestly be derived. A sentinel
#: rather than a string so the caller's branch cannot be broken by rewording.
ABSENT = object()


def benchmark_targets() -> list[str]:
    """The task config's own feature set, not a hardcoded 'pf_full'."""
    return feature_set_items(load_task_config()["benchmark"]["feature_set"])


def check_composite_ordering(universe: list[str], computed: dict) -> None:
    """Fail loudly if a composite depends on one declared after it.

    ``add_computed_features`` walks ``items`` in order, so a composite over
    another composite is only correct when its dependency comes first --
    otherwise the formula consumes whatever native column happens to be there,
    which is the S1 bug one level up.

    No such case exists today (no benchmark composite takes a composite as
    input). This does NOT quietly reorder if one appears, because ``build()``
    calls the same function with the same universe-ordered list: correcting it
    here would make this script's truth diverge from the benchmark's own
    targets, which is the exact failure the script exists to prevent. A config
    that trips this is a benchmark bug to surface, not to paper over.
    """
    pos = {t: i for i, t in enumerate(universe)}
    for name in universe:
        if name not in computed:
            continue
        for c in computed[name][0]:
            if c in computed and c in pos and pos[c] > pos[name]:
                raise Precondition(
                    f"'{name}' is computed from '{c}', which the feature set "
                    f"declares AFTER it, so add_computed_features would build "
                    f"'{name}' from a native '{c}' rather than the recomputed "
                    f"one. proforma20q build has the same behaviour -- fix the "
                    f"ordering in feature_sets.yaml rather than working around "
                    f"it here.")


def apply_compustat_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the canonical format variant, when the pull carries the columns."""
    filters = load_task_config()["universe"]["compustat_filters"]
    present = [c for c in filters if c in df.columns]
    if not present:
        say("  WARNING: none of "
            f"{sorted(filters)} present -- cannot restrict to the canonical "
            "format variant. If your pull has more than one variant per "
            "firm-quarter, which one becomes 'truth' depends on your query's "
            "row order. Re-pull with those columns, or filter them yourself.")
        return df
    if len(present) < len(filters):
        say(f"  WARNING: only {present} present of {sorted(filters)}; "
            "filtering on what is available")
    keep = np.ones(len(df), dtype=bool)
    for c in present:
        keep &= df[c].astype(str).str.strip().eq(str(filters[c])).to_numpy()
    n = int((~keep).sum())
    if n:
        say(f"  dropped {n:,} rows outside "
            + ", ".join(f"{c}={filters[c]}" for c in present))
    return df.loc[keep]


def normalize_gvkey(s: pd.Series) -> pd.Series:
    """gvkey -> zero-padded 6-character string, as the release keys it.

    Runs BEFORE de-duplication, deliberately. Padding is many-to-one ('1690' and
    '001690' are the same firm), so normalizing afterwards would let two rows for
    one firm-quarter through the (firm, quarter) de-duplication and collide only
    at the very end.
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


def resolve_targets(df: pd.DataFrame, wanted: list[str], *, universe: list[str],
                    ytd_sources: set[str], explicit: bool) -> list[str]:
    """Targets we can honestly emit.

    A computed item whose formula inputs are absent, or a YTD-derived flow whose
    ``{base}y`` source was absent, is NOT emitted -- even when a like-named
    native Compustat column would satisfy a bare ``in df.columns`` lookup.
    """
    wanted = list(dict.fromkeys(wanted))          # a repeat would duplicate rows

    # Only items in the universe were passed to add_computed_features, so only
    # those can be vouched for. _computed_definitions() is wider than the
    # benchmark -- it carries dvcq and neiq -- and one of those would otherwise
    # pass the requires-check having never been computed.
    outside = [t for t in wanted if t not in universe]
    if outside and explicit:
        raise Precondition(
            f"not benchmark targets: {sorted(outside)}. This builds truth for "
            f"the {len(universe)} items the forecasts cover.")
    wanted = [t for t in wanted if t in universe]

    computed = _computed_definitions()
    ytd_of = {f"{b}q": b for b in _ytd_bases()}
    in_universe = set(universe)

    def why_unresolvable(name: str, seen: frozenset) -> str | object | None:
        """None if ``name`` is honestly derivable, else the reason it is not.

        Recursive, because derivability is a property of the transitive input
        closure and not of a target on its own. ``fcfq = oancfq - capxq`` is the
        case that matters: both inputs are themselves YTD-derived, so a pull with
        native ``oancfq``/``capxq`` and no ``oancfy``/``capxy`` satisfies a bare
        requires-check while the formula silently consumes two columns the
        de-cumulation never produced.
        """
        if name in seen:
            return f"circular definition through {name}"
        if name in computed:
            if name not in in_universe:
                # add_computed_features was passed the universe, so an item
                # outside it was never recomputed -- any column of that name is
                # Compustat's own and means something else.
                return f"'{name}' is not a benchmark target, so it was not recomputed"
            reasons = []
            for c in computed[name][0]:
                r = why_unresolvable(c, seen | {name})
                if r:
                    reasons.append(
                        f"{c} (absent from this pull)" if r is ABSENT
                        else f"{c} ({r})")
            return "needs " + "; ".join(reasons) if reasons else None
        if name in ytd_of:
            return (None if ytd_of[name] in ytd_sources
                    else f"needs the year-to-date source '{ytd_of[name]}y'")
        return None if name in df.columns else ABSENT

    unresolvable, absent = {}, []
    for t in wanted:
        why = why_unresolvable(t, frozenset())
        if why is ABSENT:
            absent.append(t)
        elif why:
            unresolvable[t] = why

    if explicit and (unresolvable or absent):
        # Report both at once: raising on the unresolvable ones alone sends the
        # caller round the loop again to discover the merely-absent ones.
        parts = []
        for t, why in sorted(unresolvable.items()):
            # Only claim a native column exists when one actually does -- the
            # same condition the SKIP line uses.
            # Not necessarily a *Compustat* column: fcfq reaches here having
            # been built by add_computed_features from bad inputs, and Compustat
            # ships no fcfq at all. Either way the column is not the series the
            # forecasts are of.
            native = (" -- a column of that name is present and is NOT the "
                      "same series") if t in df.columns else ""
            parts.append(f"{t} {why}{native}")
        if absent:
            parts.append(f"not in this pull: {sorted(absent)}")
        raise Precondition("requested target(s) cannot be produced: "
                           + "; ".join(parts))

    for t, why in sorted(unresolvable.items()):
        native = " (a native column of that name is present and is NOT it)" \
            if t in df.columns else ""
        say(f"  SKIP {t}: {why}{native}")
    if absent:
        say(f"  {len(absent)} target(s) not in this pull: {sorted(absent)}")
    return [t for t in wanted if t not in unresolvable and t not in absent]


def build_usd_truth(raw: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Raw fundq pull -> long ``(firm_id, quarter, target, actual_musd)``."""
    explicit = targets is not None
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
        say(f"  dropped {n_null:,} rows with no gvkey")
        df = df[df["firm_id"].notna()].copy()
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.end_time

    # Filter first: rows the benchmark's universe excludes should not be able to
    # abort the run on a malformed gvkey they carry. Normalization only has to
    # precede DE-DUPLICATION, and this does.
    df = apply_compustat_filters(df)
    if df.empty:
        raise Precondition("no rows left after the canonical format filters")
    df["firm_id"] = normalize_gvkey(df["firm_id"])

    # Which YTD sources the pull actually had. convert_ytd_to_quarterly drops
    # them, and skips silently for any base it never saw.
    ytd_sources = {b for b in _ytd_bases() if f"{b}y" in df.columns}

    n0 = len(df)
    # de-cumulate BEFORE computing: fcfq = oancfq - capxq needs quarterly inputs
    df = convert_ytd_to_quarterly(df, firm_col="firm_id")
    if len(df) != n0:
        say(f"  de-duplicated {n0 - len(df):,} (firm, calendar quarter) rows")

    universe = benchmark_targets()
    check_composite_ordering(universe, _computed_definitions())
    df = add_computed_features(df, universe)

    have = resolve_targets(df, list(targets) if explicit else universe,
                           universe=universe, ytd_sources=ytd_sources,
                           explicit=explicit)
    if not have:
        raise Precondition("no targets derivable from this pull")

    fid = df["firm_id"].to_numpy()
    qtr = df["quarter"].to_numpy()

    # Melt one target at a time. This does not lower the peak much -- the
    # assembled frame and its sort dominate -- but it avoids an n x 78 spike.
    frames, empty = [], []
    for t in have:
        vals = df[t].to_numpy(dtype="float64", na_value=np.nan, copy=False)
        keep = np.isfinite(vals)
        if not keep.any():
            empty.append(t)
            continue
        frames.append(pd.DataFrame({
            "firm_id": fid[keep], "quarter": qtr[keep], "target": t,
            "actual_musd": vals[keep].astype(np.float32)}))
    if empty and explicit:
        raise Precondition(
            f"requested target(s) derivable but entirely missing in this pull: "
            f"{sorted(empty)}")
    if empty:
        say(f"  {len(empty)} target(s) entirely missing: {sorted(empty)}")
    if not frames:
        raise Precondition("every derivable target is entirely missing")

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["firm_id", "quarter", "target"], ignore_index=True)

    # Invariant: firm_id is normalized BEFORE de-duplication, so (firm_id,
    # quarter) is unique, and `have` is de-duplicated, so each target appears in
    # exactly one frame. A duplicate here means one of those two stopped holding.
    dup = int(long.duplicated(["firm_id", "quarter", "target"]).sum())
    if dup:
        raise AssertionError(
            f"{dup:,} duplicate (firm_id, quarter, target) rows after melt -- "
            f"(firm_id, quarter) uniqueness or target de-duplication broke")
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
    say(f"Reading {src}")
    if src.suffix.lower() in (".csv", ".txt", ".gz"):
        raw = pd.read_csv(src, low_memory=False)
    else:
        raw = pd.read_parquet(src, engine=args.engine)
    say(f"  {len(raw):,} rows x {len(raw.columns)} columns")

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    long = build_usd_truth(raw, targets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, engine=args.engine, compression="zstd", index=False)
    say(f"\nWrote {len(long):,} rows to {out} ({out.stat().st_size/1e6:.0f} MB)")
    say(f"  firms {long['firm_id'].nunique():,}  "
        f"quarters {long['quarter'].nunique()}  "
        f"targets {long['target'].nunique()}")
    say("\nMerge with:  forecast.target_quarter  ==  truth.quarter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
