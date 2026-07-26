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

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .build import required_raw_columns
from .config import load_task_config

# Column names are interpolated into the SELECT, so they must be plain SQL
# identifiers. The list is config-derived, which makes a typo in
# feature_sets.yaml -- not an attacker -- the realistic way something else
# arrives here; either way it must not become part of the query.
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*\Z")
# `schema.table`, and the short alphanumeric codes of the CCM link columns.
_QUALIFIED_NAME = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?\Z")
_LINK_CODE = re.compile(r"[A-Za-z0-9]{1,4}\Z")

# Canonical Compustat query filters (configs/task.yaml -> universe).
_COMPUSTAT_FILTERS = {"indfmt": "INDL", "datafmt": "STD", "consol": "C", "popsrc": "D"}
# CRSP link filters. These are part of the SAMPLE DEFINITION (the merge drops
# 6.1% of rows, and the dropped firms are ~7x smaller by median assets), so the
# authoritative copy lives in task.yaml under `universe.crsp_link`. These
# constants are the fallback for a task.yaml that predates that block.
_CCM_LINKTYPE = ("LU", "LC")
_CCM_LINKPRIM = ("P", "C")
_CCM_TABLE = "crsp.ccmxpf_lnkhist"


def crsp_link_config() -> dict:
    """The CRSP link filter, from ``task.yaml`` (falling back to the constants).

    Validated on the way out for the same reason the fundq projection is: every
    field here is interpolated into SQL, and a typo in ``task.yaml`` -- not an
    attacker -- is the realistic way something unexpected arrives. A clear config
    error beats a confusing Postgres one.
    """
    cfg = load_task_config().get("universe", {}).get("crsp_link", {}) or {}
    table = str(cfg.get("table", _CCM_TABLE))
    if not _QUALIFIED_NAME.match(table):
        raise ValueError(
            f"universe.crsp_link.table is not a plain [schema.]table name: {table!r}")
    out = {
        "table": table,
        "linktype": tuple(cfg.get("linktype", _CCM_LINKTYPE)),
        "linkprim": tuple(cfg.get("linkprim", _CCM_LINKPRIM)),
        "impose_link_dates": bool(cfg.get("impose_link_dates", True)),
    }
    for field in ("linktype", "linkprim"):
        bad = [v for v in out[field] if not _LINK_CODE.match(str(v))]
        if bad:
            raise ValueError(f"universe.crsp_link.{field} has non-code value(s): {bad}")
    return out

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


def _connect_once(wrds_username: str):
    """Open a WRDS connection with **exactly one** authentication attempt.

    ``wrds.Connection(...)`` autoconnects, and ``Connection.connect()`` retries:
    if the first attempt fails it prompts for a username/password and connects a
    second time. A second attempt means a second Duo push, and repeated attempts
    without a Duo response deactivate the account -- the one failure mode this
    package must never cause. So construct with ``autoconnect=False`` and drive
    the client's single-attempt path directly.
    """
    try:
        import wrds  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the 'wrds' package is required for downloading. Install with "
            "`pip install proforma-20q[wrds]` and configure ~/.pgpass.") from e

    print(f"Connecting to WRDS as {wrds_username} (single attempt; if this fails, "
          f"stop and check your credentials -- do NOT re-run in a loop)...")
    db = wrds.Connection(wrds_username=wrds_username, autoconnect=False)
    single = getattr(db, "_Connection__make_sa_engine_conn", None)
    if single is None:  # pragma: no cover - depends on the installed wrds version
        print("  WARNING: this 'wrds' version exposes no single-attempt connect; "
              "falling back to Connection.connect(), which may prompt and retry. "
              "If you are prompted for a password, answer nothing and abort.")
        db.connect()
    else:
        single(raise_err=True)
    return db


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


def _check_identifiers(names) -> list[str]:
    bad = [c for c in names if not _IDENTIFIER.match(str(c))]
    if bad:
        raise ValueError(
            f"refusing to build a query from non-identifier column name(s): {bad}. "
            "Check the feature-set / YTD config.")
    return list(names)


def _plain_str(s: pd.Series) -> pd.Series:
    """Normalize a key column to plain ``object`` strings.

    ``wrds.raw_sql`` returns ``string[python]`` / ``Float64`` (nullable) dtypes,
    while a chunk round-tripped through parquet comes back as ``object``. Merging
    a resumed panel against a freshly-pulled table then dies with
    ``MergeError: incompatible merge keys``. Normalizing both sides is the
    difference between resume working and resume costing another Duo prompt.
    """
    return s.astype(str).astype(object)


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
    lost: dict[str, int] = {}
    for col in df.columns:
        pg = declared.get(col, "").lower()
        if pg in _PG_NUMERIC and df[col].dtype != np.float64:
            before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif pg in _PG_DATE and not pd.api.types.is_datetime64_any_dtype(df[col]):
            before = df[col].notna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            continue
        dropped = int(before - df[col].notna().sum())
        if dropped:
            lost[col] = dropped
    if lost:
        # `errors="coerce"` turns anything unparseable into NaN. Postgres should
        # never hand back such a value, so a non-zero count means the input was
        # not what its declaration says -- report it rather than absorb it.
        print(f"  WARNING: coercion nulled values that were present: {lost}")
    return df


def _fallback_numeric_coercion(chunks: list[pd.DataFrame]) -> None:
    """Dtype repair for when the Postgres declarations are unavailable.

    Same trap, weaker evidence: a column that is numeric in ANY chunk is numeric,
    so an all-NULL year that arrived as ``object`` gets coerced instead of
    poisoning the concatenated column. In place, before the concat.
    """
    numeric: set[str] = set()
    for part in chunks:
        numeric |= {c for c in part.columns
                    if pd.api.types.is_numeric_dtype(part[c])}
    for part in chunks:
        for col in numeric:
            if col in part.columns and not pd.api.types.is_numeric_dtype(part[col]):
                part[col] = pd.to_numeric(part[col], errors="coerce")


def _year_chunks(start_year: int, end_year: int, chunk_years: int):
    if chunk_years < 1:
        # `hi = y + chunk_years - 1 < y` makes `y = hi + 1` go backwards and the
        # loop never terminates -- after authenticating.
        raise ValueError(f"chunk_years must be >= 1, got {chunk_years}")
    if start_year > end_year:
        raise ValueError(f"start_year {start_year} is after end_year {end_year}")
    y = start_year
    while y <= end_year:
        hi = min(y + chunk_years - 1, end_year)
        yield y, hi
        y = hi + 1


def _chunk_cache_name(lo: int, hi: int, wanted, filters: dict) -> str:
    """Cache filename keyed on the projection as well as the years.

    A chunk pulled under a different projection (or ``--all-columns``) is not
    interchangeable with this one: reusing it by year alone silently unions two
    schemas, leaving a column populated for some years and NaN for others.
    """
    sig = json.dumps({"cols": sorted(map(str, wanted)), "filters": filters},
                     sort_keys=True)
    return f"fundq_{lo}_{hi}__{hashlib.md5(sig.encode()).hexdigest()[:8]}.parquet"


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
            WRDS also enforces Duo 2FA, and repeated authentication attempts
            without a Duo response **deactivate the account**. This function
            makes exactly one connection attempt and never retries: the ``wrds``
            client's own ``Connection.connect()`` falls back to prompting for
            credentials and connecting a *second* time, which is precisely the
            hazard, so the single-attempt entry point is used instead. If it
            fails, stop and ask a human -- do not re-run in a loop.
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
            (default 1). Chunking yields the same row *set* as the one-shot pull
            -- ``co_industry`` and the CCM link table are fetched in full either
            way and every downstream step runs on the concatenated panel -- and
            it makes an interrupted pull resumable, which matters for a ~1 hour
            job behind a 2FA prompt. It does NOT reduce peak memory: the chunks
            are concatenated in memory, so the peak is ~2x the assembled panel
            (the 7.9x saving comes from the column projection, not from
            chunking). Pass ``None`` or 0 for a single query.
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

    # Everything that can be checked without the database is checked before the
    # database is touched: a bad argument must not cost a Duo prompt.
    if start_year > end_year:
        raise ValueError(f"start_year {start_year} is after end_year {end_year}")
    wanted = list(columns) if columns else fundq_projection(
        task["benchmark"]["feature_set"])
    if wanted != ["*"]:
        _check_identifiers(wanted)
    windows = list(_year_chunks(start_year, end_year, chunk_years)) \
        if chunk_years else [(start_year, end_year)]

    out_dir = Path(out_dir)
    intermediate_dir = Path(intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = intermediate_dir / "raw_chunks"

    db = connection
    close_db = False
    if db is None:
        db = _connect_once(wrds_username)
        close_db = True

    try:
        f = _COMPUSTAT_FILTERS
        # -- Compustat quarterly fundamentals --
        declared = _declared_dtypes(db)
        if not declared:
            print("  WARNING: comp.fundq column declarations unavailable, so chunk "
                  "dtypes cannot be taken from them. Falling back to cross-chunk "
                  "inference; verify numeric columns in the written panel.")
        if declared and wanted != ["*"]:
            missing = [c for c in wanted if c not in declared]
            if missing:
                # Dropping a projected column silently deletes a benchmark
                # feature: `{base}q` items exist only via their `{base}y` source,
                # so a missing source removes the item from the feature set and
                # changes the artifacts' shape. That is a config/schema bug.
                raise KeyError(
                    f"{len(missing)} projected column(s) absent from comp.fundq: "
                    f"{', '.join(missing)}. Fix the feature-set config, or pass "
                    "an explicit `columns=` list if this is intentional.")
        proj = "f.*" if wanted == ["*"] else ", ".join(f"f.{c}" for c in wanted)
        print(f"Downloading Compustat quarterly ({start_date}..{end_date}); "
              f"{'all' if wanted == ['*'] else len(wanted)} columns"
              + (f", {chunk_years}-year chunks" if chunk_years else ""))

        chunks = []
        for lo, hi in windows:
            cache = chunk_dir / _chunk_cache_name(lo, hi, wanted, f)
            if cache.exists():
                cached = pd.read_parquet(cache, engine="fastparquet")
                age = pd.Timestamp.fromtimestamp(cache.stat().st_mtime)
                print(f"  {lo}-{hi}: reusing {cache.name} "
                      f"(pulled {age:%Y-%m-%d}, {len(cached):,} rows)")
                chunks.append(cached)
                continue
            # ORDER BY is not cosmetic. Postgres row order is unspecified without
            # it, and the input order decides which of a duplicate firm-quarter
            # pair survives `convert_ytd_to_quarterly`'s de-duplication (pandas
            # multi-column sort_values is a stable lexsort, so ties keep input
            # order). Pinning it here is what makes that outcome reproducible --
            # run to run, and between a chunked and a one-shot pull. It also
            # makes cached chunks byte-comparable.
            part = db.raw_sql(f"""
                SELECT {proj}
                FROM comp.fundq AS f
                WHERE f.datadate BETWEEN '{lo}-01-01' AND '{hi}-12-31'
                  AND f.indfmt = '{f['indfmt']}'
                  AND f.datafmt = '{f['datafmt']}'
                  AND f.consol  = '{f['consol']}'
                  AND f.popsrc  = '{f['popsrc']}'
                ORDER BY f.gvkey, f.datadate
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
        if not declared:
            _fallback_numeric_coercion(chunks)
        if not any(len(c.columns) for c in chunks):
            raise ValueError(
                f"comp.fundq returned no columns for {start_date}..{end_date} "
                "-- check the year range and the query filters")
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
        # `raw_sql` hands back nullable `string[python]` keys; a chunk read back
        # from the cache hands back `object`. Mismatched join-key dtypes make
        # merge_asof raise, so a resumed pull would die here -- after the auth
        # and after re-pulling every uncached year.
        compustat_df["gvkey"] = _plain_str(compustat_df["gvkey"])
        naics_sic_df["gvkey"] = _plain_str(naics_sic_df["gvkey"])
        # Stable sort: the tie order of same-datadate rows differs between a
        # chunked and a one-shot pull, and it decides which of a duplicate
        # firm-quarter pair survives `convert_ytd_to_quarterly`'s de-duplication
        # (the 2-row difference measured in the audit's F26).
        compustat_df = pd.merge_asof(
            compustat_df.sort_values(["datadate"], kind="stable"),
            naics_sic_df.sort_values(["datadate"], kind="stable"),
            by="gvkey", on="datadate", direction="backward",
        )

        # -- Financial-sector exclusion (SIC 6000-6999) --
        for rng in uni["sector_exclusions"]["sic_ranges"]:
            keep = compustat_df["sich"].isna() | ~compustat_df["sich"].between(rng["start"], rng["end"])
            n0 = len(compustat_df)
            compustat_df = compustat_df[keep]
            print(f"Sector filter SIC [{rng['start']},{rng['end']}]: {n0} -> {len(compustat_df)}")

        # -- CRSP-Compustat link table --
        # Part of the sample definition: the merge below drops ~6.1% of rows and
        # the dropped firms are ~7x smaller by median assets. See
        # task.yaml -> universe.crsp_link.
        link_cfg = crsp_link_config()
        print(f"Downloading CRSP-Compustat link table ({link_cfg['table']}, "
              f"linktype {'/'.join(link_cfg['linktype'])}, "
              f"linkprim {'/'.join(link_cfg['linkprim'])})...")
        lt = "', '".join(link_cfg["linktype"])
        lp = "', '".join(link_cfg["linkprim"])
        link_df = db.raw_sql(f"""
            SELECT *
            FROM {link_cfg['table']}
            WHERE linktype IN ('{lt}') AND linkprim IN ('{lp}')
        """)
    finally:
        if close_db:
            db.close()

    # -- Merge link -> permno, imposing link date ranges --
    print("Merging link table to attach permno...")
    ccm = attach_permno(compustat_df, link_df,
                        impose_link_dates=link_cfg["impose_link_dates"])

    out_path = out_dir / "compustat_with_permno.parquet"
    ccm.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Wrote {len(ccm):,} rows -> {out_path}")
    return out_path


def attach_permno(compustat_df: pd.DataFrame, link_df: pd.DataFrame,
                  *, impose_link_dates: bool = True) -> pd.DataFrame:
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
    if impose_link_dates:
        ccm = merged[(merged["datadate"] >= merged["linkdt"])
                     & (merged["datadate"] <= merged["linkenddt"])]
    else:
        # Configurable because task.yaml is the declared source of truth for the
        # sample definition, but this is not a tuning knob: the date window is
        # what makes the merge a FILTER rather than an enrichment, and dropping
        # it yields a different, larger universe than the published one.
        print("  WARNING: universe.crsp_link.impose_link_dates is false -- link "
              "date windows are NOT applied. The resulting sample is NOT the "
              "published one and its scores are not comparable.")
        ccm = merged
    ccm = ccm.drop(columns=["linktype", "linkdt", "linkenddt", "linkprim"])
    ccm = ccm.rename(columns={"lpermno": "permno", "lpermco": "permco"})
    return ccm
