"""(Maintainer) Build the gated canonical ``column_stats`` artifact.

This is the committed record of how the published ``column_stats`` were (and can
again be) produced. ``write_checksums`` cannot do it -- it does not even write
``column_stats`` any more, precisely so that a routine repopulation cannot put
them back into the Apache-2.0 tree. It also recomputes *every* field from one
directory, while the statistics may legitimately come from a different directory
than the one whose md5 pins are being kept -- that is
exactly what happened for the published file. The canonical release build was
deleted before its statistics were captured, so its stats were computed from
the canonical R13 research artifacts (the internal builder's output off the
same panel): row-count-identical to the pins, measured to agree with this
package's builder at worst |dmean| 1e-06 (README, "Calibration, measured"),
and ``column_stats`` are order-invariant per-column aggregates, so the two
builders' differing row order is irrelevant.

The statistics are Compustat-derived, so they do NOT go into the Apache-2.0
tree. This script writes the standalone document that is published to the gated
dataset repository and fetched at runtime by
``proforma20q.checksums.load_canonical_column_stats``. The output is ignored by
git (it lands under ``build/`` by default); committing it would reintroduce
exactly the licence problem the gate exists to solve.

The build refuses to run unless the source artifacts match the pins in
``checksums.json`` on row count and regularized-column set, and it reads
``checksums.json`` without modifying it.

After running, update ``COLUMN_STATS_MD5`` in ``src/proforma20q/checksums.py``
to the digest printed below and upload the file to the dataset repository at
``checksums/canonical_column_stats.json``. The digest pin and the uploaded
bytes must move together, or every user's ``report-drift`` will fail md5
verification and correctly report NOT VERIFIED.

Usage:
    python scripts/merge_canonical_column_stats.py --source <processed_dir> \
        --note "<where the source artifacts came from and why they are canonical>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from proforma20q.checksums import (  # noqa: E402
    CHECKSUMS_PATH, COLUMN_STATS_REMOTE, column_stats,
)
from proforma20q.gated import ARTIFACTS_REPO, repo_url  # noqa: E402
from proforma20q.schema import PARQUET_ENGINE  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "build" / "canonical_column_stats.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True,
                    help="directory holding the tabular_*__<suffix>.parquet "
                         "artifacts to compute column_stats from")
    ap.add_argument("--note", required=True,
                    help="provenance note recorded as _column_stats_source: "
                         "which build these statistics come from and why it is "
                         "canonical-equivalent")
    ap.add_argument("--checksums", default=str(CHECKSUMS_PATH),
                    help="published checksums to validate the source against")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="where to write the gated artifact (git-ignored)")
    args = ap.parse_args()

    rec = json.loads(Path(args.checksums).read_text(encoding="utf-8"))
    if rec.get("_status") != "populated":
        print("checksums.json is unpopulated; run write_checksums first.",
              file=sys.stderr)
        return 2

    doc: dict = {
        "_status": "populated",
        "_schema": "proforma20q/canonical_column_stats/1",
        "suffix": rec.get("suffix"),
        "_column_stats_source": args.note,
        "artifacts": {},
    }

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
        doc["artifacts"][name] = {"n_rows": entry["n_rows"], "column_stats": stats}
        print(f"  {len(df):,} rows OK; {len(stats)} column stats", flush=True)
        del df

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2))

    digest = hashlib.md5(out_path.read_bytes()).hexdigest()
    print(f"\nwrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  md5: {digest}")
    print(f"\nNext:\n"
          f"  1. set COLUMN_STATS_MD5 = \"{digest}\" in src/proforma20q/checksums.py\n"
          f"  2. upload to {repo_url(ARTIFACTS_REPO)}\n"
          f"     at {COLUMN_STATS_REMOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
