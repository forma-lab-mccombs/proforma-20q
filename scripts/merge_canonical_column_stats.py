"""(Maintainer) Merge per-column ``column_stats`` into the shipped checksums.json.

This is the committed record of how the shipped ``column_stats`` were (and can
again be) produced. ``write_checksums`` cannot do it: it recomputes *every*
field from one directory, but the statistics may legitimately come from a
different directory than the one whose md5 pins are being kept -- that is
exactly what happened for the shipped file. The canonical release build was
deleted before its statistics were captured, so its stats were computed from
the canonical R13 research artifacts (the internal builder's output off the
same panel): row-count-identical to the pins, measured to agree with this
package's builder at worst |dmean| 1e-06 (README, "Calibration, measured"),
and ``column_stats`` are order-invariant per-column aggregates, so the two
builders' differing row order is irrelevant.

The merge refuses to run unless the source artifacts match the existing pins
on row count and regularized-column set, and it touches NOTHING but
``column_stats`` and the ``_column_stats_source`` provenance note -- md5s,
column_md5, n_rows/n_cols, download_date and task_version are preserved
byte-for-byte.

Usage:
    python scripts/merge_canonical_column_stats.py --source <processed_dir> \
        --note "<where the source artifacts came from and why they are canonical>"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from proforma20q.checksums import CHECKSUMS_PATH, column_stats  # noqa: E402
from proforma20q.schema import PARQUET_ENGINE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True,
                    help="directory holding the tabular_*__<suffix>.parquet "
                         "artifacts to compute column_stats from")
    ap.add_argument("--note", required=True,
                    help="provenance note recorded as _column_stats_source: "
                         "which build these statistics come from and why it is "
                         "canonical-equivalent")
    ap.add_argument("--out", default=str(CHECKSUMS_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    rec = json.loads(out_path.read_text(encoding="utf-8"))
    if rec.get("_status") != "populated":
        print("checksums.json is unpopulated; run write_checksums first.",
              file=sys.stderr)
        return 2

    for name, entry in rec["artifacts"].items():
        if not name.startswith("tabular_"):
            continue
        src = Path(args.source) / name
        if not src.exists():
            print(f"source artifact missing: {src}", file=sys.stderr)
            return 2
        print(f"reading {name} ...", flush=True)
        df = pd.read_parquet(src, engine=PARQUET_ENGINE)
        if len(df) != entry["n_rows"]:
            print(f"REFUSING: {name} has {len(df):,} rows, pins say "
                  f"{entry['n_rows']:,} -- this is not the canonical sample.",
                  file=sys.stderr)
            return 1
        stats = column_stats(df)
        if set(stats) != set(entry["column_md5"]):
            print(f"REFUSING: {name} regularized-column set differs from the "
                  f"pinned one ({len(stats)} vs {len(entry['column_md5'])}).",
                  file=sys.stderr)
            return 1
        entry["column_stats"] = stats
        print(f"  {len(df):,} rows OK; {len(stats)} column stats", flush=True)
        del df

    rec["_column_stats_source"] = args.note
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
