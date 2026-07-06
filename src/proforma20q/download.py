"""WRDS download for the ProForma-20Q raw panel.

Faithful port of the Forma research repo's ``data_collection/data_download.py``.
Produces ``compustat_with_permno.parquet`` -- Compustat Fundamentals Quarterly
(``comp.fundq``) with SIC/NAICS industry codes and a CRSP ``permno`` attached,
financial firms (SIC 6000-6999) excluded.

NOTHING here is distributed with the benchmark: WRDS/Compustat credentials and
license are the user's responsibility (see README). The ``wrds`` client is an
optional dependency, imported lazily so the package installs without it.

Vintage caveat: Compustat is revised. A fresh pull today will not be
bit-identical to the canonical snapshot. Record the download date, keep this
query pinned, and use ``proforma20q build --report-drift`` to quantify any
divergence from the published checksums instead of hard-failing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .config import load_task_config

# Canonical Compustat query filters (configs/task.yaml -> universe).
_COMPUSTAT_FILTERS = {"indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D"}
_CCM_LINKTYPE = ("LU", "LC")
_CCM_LINKPRIM = ("P", "C")


def download(
    wrds_username: str,
    out_dir="data/raw",
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    intermediate_dir="data",
    connection=None,
) -> Path:
    """Download the raw Compustat+permno panel from WRDS.

    Args:
        wrds_username: WRDS username. The password is read from ``~/.pgpass`` (or
            prompted by the ``wrds`` client), never passed here.
        out_dir: directory for the final ``compustat_with_permno.parquet``.
        start_year / end_year: calendar download window (defaults from
            ``task.yaml``: 1970-2024).
        intermediate_dir: directory for intermediate pulls (raw fundq, industry,
            crsp, link table) -- useful for debugging / drift analysis.
        connection: an already-open ``wrds.Connection`` (mainly for testing); if
            None one is opened and closed here.

    Returns:
        Path to the written ``compustat_with_permno.parquet``.
    """
    task = load_task_config()
    uni = task["universe"]
    start_year = start_year or uni["start_year"]
    end_year = end_year or uni["end_year"]
    start_date, end_date = f"{start_year}-01-01", f"{end_year}-12-31"

    out_dir = Path(out_dir)
    intermediate_dir = Path(intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    db = connection
    close_db = False
    if db is None:
        try:
            import wrds  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'wrds' package is required for downloading. Install with "
                "`pip install proforma-20q[wrds]` and configure ~/.pgpass.") from e
        print(f"Connecting to WRDS as {wrds_username}...")
        db = wrds.Connection(wrds_username=wrds_username)
        close_db = True

    try:
        f = _COMPUSTAT_FILTERS
        # -- Compustat quarterly fundamentals --
        print(f"Downloading Compustat quarterly ({start_date}..{end_date})...")
        compustat_df = db.raw_sql(f"""
            SELECT f.*
            FROM comp.fundq AS f
            WHERE f.datadate BETWEEN '{start_date}' AND '{end_date}'
              AND f.indfmt = '{f['indfmt']}'
              AND f.datafmt = '{f['datafmt']}'
              AND f.consol  = '{f['consol']}'
              AND f.popsrc  = '{f['popsrc']}'
        """)

        # -- Industry codes (SIC/NAICS), as-of backward merge on datadate --
        print("Downloading industry (SIC/NAICS) codes...")
        naics_sic_df = db.raw_sql(f"""
            SELECT datadate, gvkey, sich, naicsh
            FROM comp.co_industry
            WHERE consol = '{f['consol']}' AND popsrc = '{f['popsrc']}'
        """)
        compustat_df["datadate"] = pd.to_datetime(compustat_df["datadate"])
        naics_sic_df["datadate"] = pd.to_datetime(naics_sic_df["datadate"])
        compustat_df = pd.merge_asof(
            compustat_df.sort_values(["datadate"]),
            naics_sic_df.sort_values(["datadate"]),
            by="gvkey", on="datadate", direction="backward",
        )

        # -- Financial-sector exclusion (SIC 6000-6999) --
        for rng in uni["sector_exclusions"]["sic_ranges"]:
            keep = compustat_df["sich"].isna() | ~compustat_df["sich"].between(rng["start"], rng["end"])
            n0 = len(compustat_df)
            compustat_df = compustat_df[keep]
            print(f"Sector filter SIC [{rng['start']},{rng['end']}]: {n0} -> {len(compustat_df)}")

        # -- CRSP-Compustat link table --
        print("Downloading CRSP-Compustat link table...")
        lt = "', '".join(_CCM_LINKTYPE)
        lp = "', '".join(_CCM_LINKPRIM)
        link_df = db.raw_sql(f"""
            SELECT *
            FROM crsp.ccmxpf_lnkhist
            WHERE linktype IN ('{lt}') AND linkprim IN ('{lp}')
        """)
    finally:
        if close_db:
            db.close()

    # -- Merge link -> permno, imposing link date ranges --
    print("Merging link table to attach permno...")
    ccm = attach_permno(compustat_df, link_df)

    out_path = out_dir / "compustat_with_permno.parquet"
    ccm.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Wrote {len(ccm):,} rows -> {out_path}")
    return out_path


def attach_permno(compustat_df: pd.DataFrame, link_df: pd.DataFrame) -> pd.DataFrame:
    """Attach CRSP ``permno`` / ``permco`` to the Compustat panel via the CCM link
    table, imposing link date ranges.

    Faithful port of the research repo's ``data_download.py`` Step 9: a left join
    on ``gvkey`` followed by keeping only rows whose ``datadate`` falls inside the
    link's ``[linkdt, linkenddt]`` window (missing endpoints -> open on that side,
    so gvkeys with no link survive with NaN permno). Adds ``jdate`` (datadate
    month-end) and ``year``; drops the link bookkeeping columns and renames
    ``lpermno`` / ``lpermco`` -> ``permno`` / ``permco``.
    """
    compustat_df = compustat_df.copy()
    link_df = link_df.copy()
    compustat_df["gvkey"] = compustat_df["gvkey"].astype(str)
    link_df["gvkey"] = link_df["gvkey"].astype(str)
    merged = pd.merge(compustat_df, link_df, on="gvkey", how="left")
    merged["datadate"] = pd.to_datetime(merged["datadate"])
    merged["linkdt"] = pd.to_datetime(merged["linkdt"]).fillna(pd.Timestamp.min)
    merged["linkenddt"] = pd.to_datetime(merged["linkenddt"]).fillna(pd.Timestamp.max)
    merged["jdate"] = merged["datadate"] + MonthEnd(0)
    merged["year"] = merged["datadate"].dt.year
    ccm = merged[(merged["datadate"] >= merged["linkdt"]) & (merged["datadate"] <= merged["linkenddt"])]
    ccm = ccm.drop(columns=["linktype", "linkdt", "linkenddt", "linkprim"])
    ccm = ccm.rename(columns={"lpermno": "permno", "lpermco": "permco"})
    return ccm
