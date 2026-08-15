"""The ``proforma20q`` command-line interface.

    proforma20q download  --wrds-user <user>          # pull raw Compustat from WRDS
    proforma20q build     [--wrds-user <user>]        # download (if needed) + process + verify
    proforma20q baselines                             # fit reference baselines -> forecast files
    proforma20q evaluate  my_forecasts.parquet [--against baselines]
    proforma20q validate  my_forecasts.parquet        # submission-schema check
    proforma20q report-drift                          # vintage divergence vs canonical checksums

No Compustat-derived data ships with this package. The canonical regularization
statistics, the coverage mask and its row index, and the canonical per-column
drift statistics are fetched on demand from gated Hugging Face repositories
under the Forma Non-Commercial Research Licence (WRDS-Conditioned); the code
here is Apache-2.0. `validate` and `evaluate` against `examples/` need no
credentials at all; a build needs the user's own WRDS credentials (see README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_task_config


def _default_suffix(tag: str | None = None) -> str:
    from .config import CANONICAL_TAG
    task = load_task_config()
    fs = task["benchmark"]["feature_set"]
    return f"{fs}__{tag or CANONICAL_TAG}"


# --------------------------------------------------------------------------- #
# The `wrds` client surfaces a bad/absent credential as a bare EOFError from the
# input() prompt it falls back to, or as an OperationalError from psycopg2. Both
# reach the user as a traceback that says nothing about credentials.
# NOT KeyboardInterrupt: it is a BaseException, so `except Exception` never sees
# it -- and a Ctrl-C during a ~1-hour pull must not be reported as an auth
# failure, which is the one error here that carries a lockout warning.
_AUTH_ERRORS = (EOFError,)
_AUTH_HINT = (
    "WRDS authentication failed.\n"
    "  - check the username, and that your pgpass file exists and is readable:\n"
    "      Linux/macOS  ~/.pgpass                      (chmod 600)\n"
    "      Windows      %APPDATA%\\postgresql\\pgpass.conf\n"
    "  - WRDS enforces Duo 2FA even with a valid pgpass: a connection may need\n"
    "    you to approve a push.\n"
    "  - DO NOT re-run this in a loop. Repeated attempts without a Duo response\n"
    "    cause WRDS to deactivate the account. Resolve the cause, then retry once."
)


def _auth_failed(e: BaseException) -> bool:
    if isinstance(e, _AUTH_ERRORS):
        return True
    text = f"{type(e).__name__}: {e}".lower()
    return any(k in text for k in
               ("password", "authentication", "pgpass", "no such user", "role \""))


def cmd_download(args) -> int:
    from .download import download
    try:
        download(args.wrds_user, out_dir=args.out,
                 start_year=args.start_year, end_year=args.end_year,
                 intermediate_dir=args.intermediate_dir,
                 chunk_years=args.chunk_years,
                 columns=["*"] if args.all_columns else None)
    except Exception as e:  # noqa: BLE001
        if not _auth_failed(e):
            raise
        print(f"{_AUTH_HINT}\n  (underlying error: {type(e).__name__}: {e})",
              file=sys.stderr)
        return 1
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
        try:
            raw_path = download(args.wrds_user, out_dir=raw_path.parent)
        except Exception as e:  # noqa: BLE001
            if not _auth_failed(e):
                raise
            print(f"{_AUTH_HINT}\n  (underlying error: {type(e).__name__}: {e})",
                  file=sys.stderr)
            return 1

    # --reg-stats decides the TARGET SPACE, i.e. the ground truth you will be
    # scored against. Two builds off one panel differing only in this flag share
    # ~0% of target cells. It therefore has an explicit default (canonical) and
    # announces which space is in use rather than deciding silently.
    reg_stats = args.reg_stats
    if reg_stats == "estimate":
        reg_stats = None
        print("Reg-stats: RE-ESTIMATING from this panel's train split. Your targets "
              "will NOT be comparable to the published numbers -- pass "
              "--reg-stats canonical for that.")
    elif reg_stats == "canonical":
        from .config import CANONICAL_TAG, ensure_canonical_reg_stats
        from .gated import GatedArtifactError
        # `--tag` names the OUTPUT dataset; the canonical statistics are a fixed
        # published artifact. Only look for a tag-specific one if it exists,
        # otherwise pin the published R13 set -- building a differently-tagged
        # dataset in the published target space is a legitimate thing to want.
        # The statistics are Compustat-derived and therefore gated rather than
        # bundled; this downloads and md5-verifies them on first use.
        try:
            reg_stats = ensure_canonical_reg_stats(args.tag or CANONICAL_TAG)
        except GatedArtifactError as e:
            print(f"{e}\n\n"
                  f"Alternatively pass `--reg-stats estimate` to re-estimate from your\n"
                  f"own panel -- but those targets are NOT comparable to the published\n"
                  f"numbers, so `report-drift` and the leaderboard will not apply.",
                  file=sys.stderr)
            return 2
        print(f"Reg-stats: PINNED to the published canonical statistics "
              f"({Path(reg_stats).name}).")

    which = tuple(w.strip() for w in args.which.split(",") if w.strip())
    build(raw_path, out_dir=args.out, dataset_tag=args.tag, which=which,
          reg_stats=reg_stats)

    suffix = _default_suffix(args.tag)
    if args.report_drift:
        return _print_drift(args.out, suffix, args.reference)
    if not args.no_verify:
        _print_verify(args.out, suffix)
    return 0


def cmd_baselines(args) -> int:
    from .baselines import run_baselines
    suffix = _default_suffix(args.tag)
    which = [w.strip() for w in args.which.split(",")] if args.which else None
    try:
        run_baselines(args.processed, suffix, args.out, which=which)
    except FileNotFoundError as e:
        # No build present: print a clean one-liner (like build/evaluate) instead
        # of dumping a raw traceback.
        print(str(e), file=sys.stderr)
        return 2
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

    res = evaluate_forecasts(forecasts, truth, allow_missing=args.allow_missing,
                             sample_mask=args.sample_mask, grid_rows=args.grid_rows)
    print("\n=== Leaderboard (global pool) ===")
    print(res.leaderboard(args.sort).to_string(index=False))
    if args.out:
        written = res.write_csvs(args.out)
        print(f"\nWrote {len(written)} metric CSV(s) to {args.out}")
    return 0 if res.invariant_ok else 1


def cmd_validate(args) -> int:
    from .schema import scan_forecast_file
    # Streamed by row-group: a full-coverage submission is ~550M rows / ~73 GB
    # as a frame, so validation cannot start by reading the file. One pass, not
    # one per kind of finding.
    try:
        problems, n_rows, warnings = scan_forecast_file(args.forecast)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to read {args.forecast}: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"  WARNING: {w}", file=sys.stderr)
    if not problems:
        print(f"OK: {args.forecast} conforms to the submission schema ({n_rows:,} rows)"
              + (f", with {len(warnings)} warning(s)." if warnings else "."))
        return 0
    print(f"INVALID: {args.forecast}", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


def cmd_report_drift(args) -> int:
    return _print_drift(args.processed, _default_suffix(args.tag), args.reference)


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
    if not rep["all_match"]:
        # Expected on any fresh WRDS pull (Compustat is revised; see README).
        # Without this pointer, the last thing a ~50-minute build prints is a
        # wall of "mismatch" with no verdict.
        print("  NOTE: md5 mismatches are EXPECTED on a fresh WRDS pull -- "
              "Compustat is revised, so\n  bit-exact equality with the canonical "
              "snapshot is not achievable and not the target.\n  Run "
              "`proforma20q report-drift` for the calibrated PASS/FAIL verdict.")


def _print_drift(processed_dir, suffix, reference=None) -> int:
    from .checksums import reference_record, report_drift

    published = reference_record(reference, suffix) if reference else None
    rep = report_drift(processed_dir, suffix, published)
    artifacts = {k: v for k, v in rep.items() if not k.startswith("_")}
    # A published artifact with no local file reads as "missing"; anything else
    # is a file that exists on disk. Zero local files == nothing was built, which
    # must be an explicit error, not a green (empty) "no drift" report.
    present = [name for name, r in artifacts.items() if r.get("status") != "missing"]
    if not present:
        if rep.get("_note") and not artifacts:
            print(f"\nDrift report: {rep['_note']}")
            return 0
        print(f"no built artifacts found under {processed_dir} -- run `proforma20q build` "
              f"first", file=sys.stderr)
        return 2

    against = f"reference build {reference}" if reference else "canonical checksums"
    print(f"\n=== Vintage-drift report (vs {against}) ===")
    for name, r in artifacts.items():
        status = r.get("status")
        if status in ("missing", "extra"):
            print(f"  {status:>9}  {name}")
            continue
        if "n_cols_over_threshold" in r:
            verdict = "PASS" if r["pass"] else "FAIL"
            print(f"  [{verdict}] {name}")
            print(f"      rows {r['row_count_delta']:+,} ({r['row_count_delta_frac']:.3%}); "
                  f"{r['n_cols_over_threshold']}/{r['n_cols']} columns beyond tolerance")
            if r["n_cols"] == 0:
                print("      NO COLUMNS IN COMMON -- nothing was compared; this is not a pass")
            if r["n_cols_only_here"] or r["n_cols_only_published"]:
                print(f"      column set differs: {r['n_cols_only_here']} only here "
                      f"({', '.join(r['cols_only_here'][:4])}...), "
                      f"{r['n_cols_only_published']} only in the reference "
                      f"({', '.join(r['cols_only_published'][:4])}...)")
            print(f"      worst |delta mean| {r['worst_abs_mean_delta']:.2e} "
                  f"({r['worst_abs_mean_delta_col']}), "
                  f"|delta sd| {r['worst_abs_sd_delta']:.2e}, "
                  f"|delta coverage| {r['worst_abs_coverage_delta']:.2e}; "
                  f"median |delta mean| {r['median_abs_mean_delta']:.2e}")
            if r["cols_over_threshold"]:
                print(f"      first offenders: {', '.join(r['cols_over_threshold'])}")
        elif "frac_diff" in r:
            print(f"  {name}: {r['n_diff']}/{r['n_cols']} cols diverge "
                  f"({r['frac_diff']:.1%}) -- hash metric, not a drift measure; "
                  f"rows {r.get('row_count_delta', 0):+,}")
        else:
            print(f"  {name}: file md5 match={r.get('file_md5_match')}")

    idm = rep.get("_id_maps")
    if idm:
        print("\n  --- id maps (tuple-view embedding orderings) ---")
        for name in ("account_id_map", "industry_id_map"):
            r = idm[name]
            if r["status"] == "ok":
                print(f"  [PASS] {name}: {r['n']} entries match the pinned "
                      f"canonical reference exactly")
            elif r["status"] == "no_reference":
                print(f"  {name}: no pinned reference for this suffix (skipped)")
            elif r["status"] == "missing":
                print(f"  [FAIL] {name}: not found in the build")
            else:
                print(f"  [FAIL] {name}: DIFFERS from the pinned canonical "
                      f"reference ({r['n_built']} vs {r['n_reference']} entries) "
                      f"-- an embedding trained under this map is permuted "
                      f"relative to the canonical ordering")
                for d in r.get("first_diffs", []):
                    print(f"      {d}")
        f = idm["firm_id_map"]
        if f["status"] == "missing":
            print("  [FAIL] firm_id_map: not found in the build")
        else:
            verdict = "PASS" if f["status"] == "ok" else "FAIL"
            if f["delta_frac"] is not None:
                vs_ref = (f"vs {f['n_reference']:,} reference "
                          f"({f['delta_frac']:.3%} delta; gvkey universe drifts "
                          f"by vintage, ids must be re-mapped via gvkey strings, "
                          f"never assumed positionally)")
            else:
                vs_ref = "(no reference count to compare)"
            print(f"  [{verdict}] firm_id_map: {f['n_firms']:,} firms {vs_ref}; "
                  f"ordering rule (sorted gvkeys, ids 0..n-1): "
                  f"{'ok' if f['ordering_rule_ok'] else 'VIOLATED'}")

    if rep.get("_column_stats_source"):
        print(f"\n  reference statistics provenance: {rep['_column_stats_source']}")
    if rep.get("_note"):
        print(f"\n  NOTE: {rep['_note']}")
    if rep.get("_not_verified"):
        # Nothing was compared. Say so in those words and exit non-zero: the
        # published promise is "verified by published checksums", and a
        # reference that could not be reached means UNVERIFIED, not "fine".
        print("\n  VERDICT: NOT VERIFIED")
        print("  The canonical per-column statistics could not be fetched, or do\n"
              "  not apply to this build, so no drift comparison was performed.\n"
              "  This is NOT a pass -- your build has not been checked against the\n"
              "  published reference.\n")
        for line in rep["_not_verified"].splitlines():
            print(f"  {line}")
        print("\n  Or compare against a build you already trust:\n"
              "    proforma20q report-drift --reference <dir>")
        return 3
    if rep.get("_pass") is None:
        print("  VERDICT: indeterminate (no comparable statistic published)")
        return 0
    th = rep["_thresholds"]
    print(f"\n  thresholds: rows <= {th['max_row_delta_frac']:.1%}, "
          f"per-column |delta mean| <= {th['max_abs_mean_delta']}, "
          f"|delta sd| <= {th['max_abs_sd_delta']}, "
          f"|delta coverage| <= {th['max_abs_coverage_delta']}")
    print(f"  VERDICT: {'PASS' if rep['_pass'] else 'FAIL'}")
    return 0 if rep["_pass"] else 1


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
    d.add_argument("--intermediate-dir", default="data",
                   help="where per-chunk fundq parquets are cached (default: data/)")
    d.add_argument("--chunk-years", type=int, default=1,
                   help="pull comp.fundq in N-year chunks (default 1; 0 = one query). "
                        "Chunks are cached under <intermediate-dir>/raw_chunks so an "
                        "interrupted pull resumes")
    d.add_argument("--all-columns", action="store_true",
                   help="SELECT f.* instead of the 82-column projection "
                        "(~7.9x more data; ~100 GB peak over 1970-2024)")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="download (if needed) + process + verify checksums")
    b.add_argument("--wrds-user", default=None, help="download raw first if not present")
    b.add_argument("--raw", default="data/raw/compustat_with_permno.parquet")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--tag", default=None, help="dataset tag (default: canonical R13 tag)")
    b.add_argument("--reg-stats", default="canonical",
                   help="which regularization statistics define the TARGET SPACE. "
                        "'canonical' (default) pins the published R13 statistics, so your "
                        "targets match the leaderboard's; 'estimate' re-estimates them from "
                        "your own train split, which produces a DIFFERENT ground truth and "
                        "non-comparable scores; or a path to a regularization_stats parquet")
    b.add_argument("--which", default="tabular,tuple")
    b.add_argument("--report-drift", action="store_true",
                   help="quantify divergence from the canonical checksums instead of bit-verify; "
                        "exits non-zero if the divergence exceeds the published thresholds")
    b.add_argument("--reference", default=None,
                   help="compare against a reference build directory instead of the "
                        "published checksums")
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
    e.add_argument("--sample-mask", default=None,
                   help="restrict scoring to a pooled-sample mask (parquet of firm/target/"
                        "origin/horizon keys, or the grid-aligned .npy bit array), e.g. "
                        "the paper's Full 327.2M-cell sample")
    e.add_argument("--grid-rows", default=None,
                   help="canonical row index (full_sample_grid_rows.parquet) that lets a "
                        "grid-aligned --sample-mask apply to a vintage-drifted rebuild; "
                        "the bitmap is realigned by (firm, origin) value")
    e.set_defaults(func=cmd_evaluate)

    v = sub.add_parser("validate", help="check a forecast file against the submission schema")
    v.add_argument("forecast")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("report-drift", help="vintage divergence of a build vs canonical checksums")
    r.add_argument("--processed", default="data/processed")
    r.add_argument("--tag", default=None)
    r.add_argument("--reference", default=None,
                   help="compare against a reference build directory instead of the "
                        "published checksums")
    r.set_defaults(func=cmd_report_drift)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
