"""The ``proforma20q`` command-line interface.

    proforma20q download  --wrds-user <user>          # pull raw Compustat from WRDS
    proforma20q build     [--wrds-user <user>]        # download (if needed) + process + verify
    proforma20q baselines                             # fit reference baselines -> forecast files
    proforma20q evaluate  my_forecasts.parquet [--against baselines]
    proforma20q validate  my_forecasts.parquet        # submission-schema check
    proforma20q report-drift                          # vintage divergence vs canonical checksums

Nothing WRDS-derived ships with the package; a build needs the user's own WRDS
credentials (see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_task_config


def _default_suffix(tag: str | None = None) -> str:
    task = load_task_config()
    fs = task["benchmark"]["feature_set"]
    tag = tag or "r13_node_optionD_indfe_val8"
    return f"{fs}__{tag}"


# --------------------------------------------------------------------------- #
def cmd_download(args) -> int:
    from .download import download
    download(args.wrds_user, out_dir=args.out,
             start_year=args.start_year, end_year=args.end_year)
    return 0


def cmd_build(args) -> int:
    from .build import build

    raw_path = Path(args.raw)
    if not raw_path.exists():
        if not args.wrds_user:
            print(f"Raw panel not found at {raw_path}. Provide --wrds-user to download it "
                  f"first, or point --raw at an existing compustat_with_permno.parquet.",
                  file=sys.stderr)
            return 2
        from .download import download
        raw_path = download(args.wrds_user, out_dir=raw_path.parent)

    reg_stats = args.reg_stats
    if reg_stats == "canonical":
        from .config import canonical_reg_stats_path
        reg_stats = canonical_reg_stats_path(args.tag or "r13_node_optionD_indfe_val8")

    which = tuple(w.strip() for w in args.which.split(",") if w.strip())
    build(raw_path, out_dir=args.out, dataset_tag=args.tag, which=which,
          reg_stats=reg_stats)

    suffix = _default_suffix(args.tag)
    if args.report_drift:
        _print_drift(args.out, suffix)
    elif not args.no_verify:
        _print_verify(args.out, suffix)
    return 0


def cmd_baselines(args) -> int:
    from .baselines import run_baselines
    suffix = _default_suffix(args.tag)
    which = [w.strip() for w in args.which.split(",")] if args.which else None
    run_baselines(args.processed, suffix, args.out, which=which)
    return 0


def cmd_evaluate(args) -> int:
    import pandas as pd

    from .baselines.common import load_tabular
    from .evaluate import evaluate_forecasts
    from .schema import PARQUET_ENGINE

    suffix = _default_suffix(args.tag)
    # ground truth = the tabular_test artifact (unless an explicit --truth is given)
    truth_path = Path(args.truth) if args.truth else Path(args.processed) / f"tabular_test__{suffix}.parquet"
    if not truth_path.exists():
        print(f"Ground truth not found at {truth_path}. Run `proforma20q build` first, "
              f"or pass --truth.", file=sys.stderr)
        return 2
    truth = load_tabular(truth_path)

    forecasts: dict[str, object] = {}
    for f in args.forecasts:
        forecasts[Path(f).stem.replace("__predictions", "")] = Path(f)
    if args.against == "baselines":
        fc_dir = Path(args.baseline_dir)
        for name in ("naive", "fade", "elasticnet", "linear"):
            p = fc_dir / f"{name}__predictions.parquet"
            if p.exists():
                forecasts.setdefault(name, p)
    if not forecasts:
        print("No forecast files to evaluate.", file=sys.stderr)
        return 2

    res = evaluate_forecasts(forecasts, truth, allow_missing=args.allow_missing)
    print("\n=== Leaderboard (global pool) ===")
    print(res.leaderboard(args.sort).to_string(index=False))
    if args.out:
        written = res.write_csvs(args.out)
        print(f"\nWrote {len(written)} metric CSV(s) to {args.out}")
    return 0 if res.invariant_ok else 1


def cmd_validate(args) -> int:
    from .schema import SubmissionError, read_forecast, validate_forecast
    try:
        df = read_forecast(args.forecast, validate=False)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to read {args.forecast}: {e}", file=sys.stderr)
        return 2
    problems = validate_forecast(df, strict=False)
    if not problems:
        print(f"OK: {args.forecast} conforms to the submission schema ({len(df):,} rows).")
        return 0
    print(f"INVALID: {args.forecast}", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


def cmd_report_drift(args) -> int:
    _print_drift(args.processed, _default_suffix(args.tag))
    return 0


# --------------------------------------------------------------------------- #
def _print_verify(processed_dir, suffix) -> None:
    from .checksums import verify_checksums
    rep = verify_checksums(processed_dir, suffix)
    if rep.get("_note"):
        print(f"\nChecksum verify: {rep['_note']} (skipping).")
        return
    print("\n=== Checksum verification (bit-exact) ===")
    for name, r in rep.items():
        if name == "all_match":
            continue
        print(f"  {r.get('status', '?'):>9}  {name}")
    print(f"  ALL MATCH: {rep['all_match']}")


def _print_drift(processed_dir, suffix) -> None:
    from .checksums import report_drift
    rep = report_drift(processed_dir, suffix)
    if rep.get("_note"):
        print(f"\nDrift report: {rep['_note']}")
        return
    print("\n=== Vintage-drift report (vs canonical checksums) ===")
    for name, r in rep.items():
        if "frac_diff" in r:
            print(f"  {name}: {r['n_diff']}/{r['n_cols']} cols diverge "
                  f"({r['frac_diff']:.1%}); file md5 match={r['file_md5_match']}; "
                  f"row delta={r['row_count_delta']}")
        else:
            print(f"  {name}: file md5 match={r.get('file_md5_match')}")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proforma20q", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"proforma-20q {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download raw Compustat panel from WRDS")
    d.add_argument("--wrds-user", required=True)
    d.add_argument("--out", default="data/raw")
    d.add_argument("--start-year", type=int, default=None)
    d.add_argument("--end-year", type=int, default=None)
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="download (if needed) + process + verify checksums")
    b.add_argument("--wrds-user", default=None, help="download raw first if not present")
    b.add_argument("--raw", default="data/raw/compustat_with_permno.parquet")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--tag", default=None, help="dataset tag (default: canonical R13 tag)")
    b.add_argument("--reg-stats", default=None,
                   help="normalize against a FROZEN regularization_stats parquet instead of "
                        "re-estimating (pins the target/eval space across Compustat vintages); "
                        "pass 'canonical' to use the bundled published R13 reg-stats")
    b.add_argument("--which", default="tabular,tuple")
    b.add_argument("--report-drift", action="store_true",
                   help="quantify divergence from the canonical checksums instead of bit-verify")
    b.add_argument("--no-verify", action="store_true")
    b.set_defaults(func=cmd_build)

    bl = sub.add_parser("baselines", help="fit reference baselines -> forecast parquets")
    bl.add_argument("--processed", default="data/processed")
    bl.add_argument("--out", default="results/forecasts")
    bl.add_argument("--tag", default=None)
    bl.add_argument("--which", default=None, help="comma list; default all four")
    bl.set_defaults(func=cmd_baselines)

    e = sub.add_parser("evaluate", help="score forecast file(s) against the test truth")
    e.add_argument("forecasts", nargs="*", help="forecast parquet path(s)")
    e.add_argument("--processed", default="data/processed")
    e.add_argument("--truth", default=None, help="explicit truth parquet (default: tabular_test)")
    e.add_argument("--tag", default=None)
    e.add_argument("--against", choices=["baselines"], default=None,
                   help="also score the reference baselines from --baseline-dir")
    e.add_argument("--baseline-dir", default="results/forecasts")
    e.add_argument("--out", default=None, help="write per-level metric CSVs here")
    e.add_argument("--sort", default="r2", help="leaderboard sort metric")
    e.add_argument("--allow-missing", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    v = sub.add_parser("validate", help="check a forecast file against the submission schema")
    v.add_argument("forecast")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("report-drift", help="vintage divergence of a build vs canonical checksums")
    r.add_argument("--processed", default="data/processed")
    r.add_argument("--tag", default=None)
    r.set_defaults(func=cmd_report_drift)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
