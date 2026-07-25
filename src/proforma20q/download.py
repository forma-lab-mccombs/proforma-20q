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

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .build import required_raw_columns
from .config import load_task_config

# Canonical Compustat query filters (configs/task.yaml -> universe).
_COMPUSTAT_FILTERS = {"indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D"}
_CCM_LINKTYPE = ("LU", "LC")
_CCM_LINKPRIM = ("P", "C")

# Postgres declared type -> the dtype a fully-populated column would arrive as.
_PG_NUMERIC = ("double precision", "real", "numeric", "decimal",
               "integer", "bigint", "smallint")
_PG_DATE = ("date", "timestamp without time zone", "timestamp with time zone")


def fundq_projection(feature_set: str | None = None) -> list[str]:
    """The ``comp.fundq`` columns the benchmark consumes (82 of the table's 648).

    ``SELECT f.*`` over 1970-2024 needs ~100 GB peak and cannot complete; the
    projection is ~7.9x smaller. Older rows cost ~3.5x more per row than recent
    ones (early columns are mostly NULL and come back as ``object``), so a job
    sized from a recent slice badly understates the full pull.
    """
    return required_raw_columns(feature_set, include_attached=False,
                                prefer_ytd_source=True)


def _declared_dtypes(db, table: str = "fundq", schema: str = "comp") -> dict[str, str]:
    """Column -> Postgres declared type for a WRDS table ({} if unavailable)."""
    try:
        decl = db.raw_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = '{schema}' AND table_name = '{table}'")
    except Exception as e:  # noqa: BLE001 - purely advisory
        print(f"  (could not read {schema}.{table} column declarations: {e})")
        return {}
    return dict(zip(decl["column_name"].astype(str), decl["data_type"].astype(str)))


def coerce_declared_dtypes(df: pd.DataFrame, declared: dict[str, str]) -> pd.DataFrame:
    """Cast columns to the dtype their Postgres declaration implies.

    **The chunking trap.** A column that is entirely NULL for one year comes back
    as ``object``; concatenating that chunk with years where the same column is
    numeric produces an ``object`` column, and a later ``astype(str)`` or parquet
    round-trip silently stringifies genuinely numeric data (this bit the audit on
    ``oancfy``, a required YTD source). Take dtypes from the *declarations*, not
    from the chunks -- per chunk, before the concat.
    """
    if not declared:
        return df
    for col in df.columns:
        pg = declared.get(col, "").lower()
        if pg in _PG_NUMERIC and df[col].dtype != np.float64:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif pg in _PG_DATE and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _year_chunks(start_year: int, end_year: int, chunk_years: int):
    y = start_year
    while y <= end_year:
        hi = min(y + chunk_years - 1, end_year)
        yield y, hi
        y = hi + 1


def download(
    wrds_username: str,
    out_dir="data/raw",
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    intermediate_dir="data",
    connection=None,
    chunk_years: int | None = 1,
    columns: list[str] | None = None,
) -> Path:
    """Download the raw Compustat+permno panel from WRDS.

    Args:
        wrds_username: WRDS username. The password is read from ``~/.pgpass`` (or
            ``%APPDATA%\\postgresql\\pgpass.conf`` on Windows), never passed here.
            WRDS also enforces Duo 2FA; **never retry authentication in a loop**
            -- repeated attempts without a Duo response deactivate the account.
        out_dir: directory for the final ``compustat_with_permno.parquet``.
        start_year / end_year: calendar download window (defaults from
            ``task.yaml``: 1970-2024).
        intermediate_dir: directory for intermediate pulls. Per-chunk fundq
            parquets land in ``<intermediate_dir>/raw_chunks/`` and are reused on
            a re-run, so an interrupted pull resumes instead of restarting.
        connection: an already-open ``wrds.Connection`` (mainly for testing); if
            None one is opened and closed here. One connection is opened per
            call and reused for every chunk.
        chunk_years: pull ``comp.fundq`` in windows of this many calendar years
            (default 1). Chunking is provably equivalent to the one-shot pull --
            ``co_industry`` and the CCM link table are fetched in full either way
            and every downstream step runs on the concatenated panel -- and it
            keeps peak memory flat over the 55-year window. Pass ``None`` or 0
            for a single query.
        columns: explicit ``comp.fundq`` projection; defaults to
            :func:`fundq_projection` (the 82 columns the benchmark consumes).
            Pass ``["*"]`` for the old ``SELECT f.*`` behaviour (~100 GB peak
            over 1970-2024 -- it does not complete on an ordinary machine).

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
    chunk_dir = intermediate_dir / "raw_chunks"

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
        wanted = list(columns) if columns else fundq_projection(
            task["benchmark"]["feature_set"])
        declared = _declared_dtypes(db)
        if declared and wanted != ["*"]:
            missing = [c for c in wanted if c not in declared]
            if missing:
                print(f"  note: {len(missing)} projected column(s) absent from "
                      f"comp.fundq, skipping: {', '.join(missing)}")
                wanted = [c for c in wanted if c in declared]
        proj = "f.*" if wanted == ["*"] else ", ".join(f"f.{c}" for c in wanted)
        print(f"Downloading Compustat quarterly ({start_date}..{end_date}); "
              f"{'all' if wanted == ['*'] else len(wanted)} columns"
              + (f", {chunk_years}-year chunks" if chunk_years else ""))

        windows = list(_year_chunks(start_year, end_year, chunk_years)) \
            if chunk_years else [(start_year, end_year)]
        chunks = []
        for lo, hi in windows:
            cache = chunk_dir / f"fundq_{lo}_{hi}.parquet"
            if cache.exists():
                print(f"  {lo}-{hi}: reusing {cache.name}")
                chunks.append(pd.read_parquet(cache, engine="fastparquet"))
                continue
            part = db.raw_sql(f"""
                SELECT {proj}
                FROM comp.fundq AS f
                WHERE f.datadate BETWEEN '{lo}-01-01' AND '{hi}-12-31'
                  AND f.indfmt = '{f['indfmt']}'
                  AND f.datafmt = '{f['datafmt']}'
                  AND f.consol  = '{f['consol']}'
                  AND f.popsrc  = '{f['popsrc']}'
            """)
            # Per chunk, BEFORE the concat: a column that is all-NULL in one year
            # arrives as `object` and would poison the concatenated column.
            part = coerce_declared_dtypes(part, declared)
            if len(windows) > 1:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".parquet.tmp")
                part.to_parquet(tmp, index=False, engine="fastparquet")
                tmp.replace(cache)
            print(f"  {lo}-{hi}: {len(part):,} rows")
            chunks.append(part)
        compustat_df = chunks[0] if len(chunks) == 1 else \
            pd.concat(chunks, ignore_index=True)
        del chunks
        # A cached chunk round-trips through parquet, so re-assert the declared
        # dtypes on the assembled panel too.
        compustat_df = coerce_declared_dtypes(compustat_df, declared)
        print(f"Compustat quarterly: {len(compustat_df):,} rows x "
              f"{compustat_df.shape[1]} columns")

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
