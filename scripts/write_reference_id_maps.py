"""(Maintainer) Write the pinned canonical id-map CSVs into ``reference/``.

The tuple view's account and industry integer ids are embedding indices, and
both maps are **data-independent**: the account universe is the feature set's
statement items plus ``scale`` (ids by ``sorted()``, from the task config) and
the industry universe is the pinned FF48 table (``ff48_sic_ranges.json``).
Regenerating them here therefore reproduces the canonical build's orderings
exactly, on any machine, with no WRDS data. The firm map is deliberately NOT
pinned -- its gvkey universe drifts with the Compustat vintage; see
``build.verify_id_maps``.

Byte-for-byte identical to what ``proforma20q build`` writes for the canonical
feature set (same table builders, same ``to_csv``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proforma20q.build import (  # noqa: E402
    account_id_map_table,
    canonical_account_id_map,
    industry_id_map_table,
)
from proforma20q.config import canonical_id_map_path  # noqa: E402


def main() -> int:
    tables = {
        "account_id_map": account_id_map_table(canonical_account_id_map()),
        "industry_id_map": industry_id_map_table(),
    }
    for name, table in tables.items():
        out = canonical_id_map_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"wrote {out}  ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
