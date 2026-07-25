"""Unit tests for the WRDS-free parts of download (the CCM link merge).

No WRDS connection: attach_permno is pure pandas over a Compustat panel and a
CCM link table, so the port's permno-attachment (date-range imposition, NaN link
handling, column rename/drop) is fully testable on synthetic frames. The query
shape (column projection, year chunking, dtype handling) is covered against a
stub connection that answers `raw_sql` from synthetic frames.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from proforma20q.download import (attach_permno, coerce_declared_dtypes, download,
                                  fundq_projection)


def _link_row(gvkey, lpermno, lpermco, dt, enddt):
    return {"gvkey": gvkey, "linkprim": "P", "liid": "01", "linktype": "LC",
            "lpermno": lpermno, "lpermco": lpermco,
            "linkdt": pd.Timestamp(dt), "linkenddt": pd.Timestamp(enddt)}


def test_attach_permno_date_range_and_rename():
    comp = pd.DataFrame({
        "gvkey": ["001", "001", "002", "003"],
        "datadate": pd.to_datetime(["2000-03-31", "2010-03-31", "2005-06-30", "2001-09-30"]),
        "atq": [1.0, 2.0, 3.0, 4.0],
    })
    link = pd.DataFrame([
        _link_row("001", 10001, 20001, "1999-01-01", "2005-12-31"),
        _link_row("002", 10002, 20002, "1990-01-01", "2024-12-31"),
    ])
    out = attach_permno(comp, link)

    assert "permno" in out.columns and "permco" in out.columns
    assert "lpermno" not in out.columns
    for dropped in ("linktype", "linkdt", "linkenddt", "linkprim"):
        assert dropped not in out.columns
    assert "jdate" in out.columns and "year" in out.columns

    by_key = {(r.gvkey, r.datadate): r for r in out.itertuples()}
    # 001 @2000 is inside the link window -> permno attached
    assert by_key[("001", pd.Timestamp("2000-03-31"))].permno == 10001
    # 001 @2010 is PAST linkenddt (2005) -> row dropped
    assert ("001", pd.Timestamp("2010-03-31")) not in by_key
    # 002 inside its (wide) window
    assert by_key[("002", pd.Timestamp("2005-06-30"))].permno == 10002


def test_attach_permno_unmatched_gvkey_survives_with_nan():
    comp = pd.DataFrame({
        "gvkey": ["999"],
        "datadate": pd.to_datetime(["2015-12-31"]),
        "atq": [5.0],
    })
    link = pd.DataFrame([_link_row("001", 10001, 20001, "1990-01-01", "2020-12-31")])
    out = attach_permno(comp, link)
    # gvkey with no link row: filled endpoints (min/max) keep it, permno is NaN.
    assert len(out) == 1
    assert np.isnan(out.iloc[0]["permno"])


def test_attach_permno_year_and_jdate_derived():
    comp = pd.DataFrame({"gvkey": ["001"], "datadate": pd.to_datetime(["2003-02-15"]), "atq": [1.0]})
    link = pd.DataFrame([_link_row("001", 10001, 20001, "1990-01-01", "2020-12-31")])
    out = attach_permno(comp, link).iloc[0]
    assert out["year"] == 2003
    assert out["jdate"] == pd.Timestamp("2003-02-28")


# --------------------------------------------------------------------------- #
# Query shape: projection, chunking, dtypes (I-2)
# --------------------------------------------------------------------------- #
class _StubWRDS:
    """Answers ``raw_sql`` from synthetic frames and records every query.

    ``null_year`` returns ``oancfy`` as an all-NULL object column for that one
    year -- the concatenation trap: taken from the chunk rather than from the
    Postgres declaration, that dtype poisons the whole column.
    """

    def __init__(self, declare=True, null_year=None):
        self.queries: list[str] = []
        self.declare = declare
        self.null_year = null_year
        self.closed = False

    @staticmethod
    def _declarations():
        text = {"gvkey", "tic", "conm", "indfmt", "datafmt", "consol", "popsrc"}
        rows = []
        for c in fundq_projection():
            if c in text:
                dt = "character varying"
            elif c == "datadate":
                dt = "date"
            else:
                dt = "double precision"
            rows.append({"column_name": c, "data_type": dt})
        return pd.DataFrame(rows)

    def _fundq(self, sql):
        cols = [c.strip()[2:] for c in
                re.search(r"SELECT (.+?)\s+FROM", sql, re.S).group(1).split(",")]
        lo, hi = (int(y) for y in re.findall(r"'(\d{4})-\d\d-\d\d'", sql)[:2])
        recs = []
        for year in range(lo, hi + 1):
            for q in range(1, 5):
                for gv in ("001000", "001001"):
                    r = {c: float(year + q) for c in cols}
                    r.update(gvkey=gv, tic="T" + gv, conm="Firm " + gv,
                             indfmt="INDL", datafmt="STD", consol="C", popsrc="D",
                             datadate=(pd.Timestamp(year=year, month=3 * q, day=1)
                                       + pd.offsets.MonthEnd(0)),
                             fyearq=float(year), fqtr=float(q))
                    if self.null_year is not None and year == self.null_year:
                        r["oancfy"] = None
                    recs.append(r)
        return pd.DataFrame(recs, columns=cols)

    def raw_sql(self, sql):
        self.queries.append(sql)
        if "information_schema" in sql:
            if not self.declare:
                raise RuntimeError("no information_schema access")
            return self._declarations()
        if "comp.fundq" in sql:
            return self._fundq(sql)
        if "co_industry" in sql:
            return pd.DataFrame({
                "datadate": pd.to_datetime(["1969-12-31", "1969-12-31"]),
                "gvkey": ["001000", "001001"],
                "sich": [3500.0, 2000.0], "naicsh": [334111.0, 311111.0]})
        if "ccmxpf_lnkhist" in sql:
            return pd.DataFrame([
                _link_row(g, 10000 + i, 20000 + i, "1950-01-01", "2030-01-01")
                for i, g in enumerate(("001000", "001001"))])
        raise AssertionError("unexpected query: " + sql)

    def close(self):
        self.closed = True

    def fundq_queries(self):
        return [q for q in self.queries if "comp.fundq" in q]


def test_fundq_projection_is_the_consumed_columns_only():
    """``SELECT f.*`` pulls 648 columns for the 82 the benchmark reads. The
    projection is derived from the configs, so it must contain every item the
    build consumes -- and must NOT contain the ``{base}q`` forms that do not
    exist in fundq (they are de-cumulated from ``{base}y``)."""
    proj = fundq_projection()
    assert len(proj) == len(set(proj))
    for needed in ("atq", "ltq", "seqq", "niq", "revtq", "cogsq", "xsgaq", "xrdq",
                   "gvkey", "datadate", "fyearq", "fqtr"):
        assert needed in proj, needed
    assert "oancfy" in proj and "capxy" in proj           # YTD sources
    assert "oancfq" not in proj and "capxq" not in proj   # derived, not columns
    for computed in ("gpq", "fcfq", "wcapq", "aoq_ex_intanq"):
        assert computed not in proj                       # formulas, not columns
    assert len(proj) < 100


def test_download_projects_columns_and_chunks_by_year(tmp_path):
    db = _StubWRDS()
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1998,
                   intermediate_dir=tmp_path, connection=db)
    fq = db.fundq_queries()
    assert len(fq) == 3                       # one query per year
    assert "f.*" not in " ".join(fq)          # projected, not SELECT f.*
    assert "f.atq" in fq[0] and "f.oancfy" in fq[0]
    assert not db.closed                      # a caller-supplied connection stays open

    panel = pd.read_parquet(out, engine="fastparquet")
    assert len(panel) == 3 * 4 * 2            # 3 years x 4 quarters x 2 firms
    assert set(panel["gvkey"]) == {"001000", "001001"}
    assert "permno" in panel.columns and panel["permno"].notna().all()
    cached = sorted(p.name for p in (tmp_path / "raw_chunks").glob("*.parquet"))
    assert cached == ["fundq_1996_1996.parquet", "fundq_1997_1997.parquet",
                      "fundq_1998_1998.parquet"]


def test_download_reuses_cached_chunks(tmp_path):
    kw = dict(out_dir=tmp_path / "raw", start_year=1996, end_year=1997,
              intermediate_dir=tmp_path)
    download("tester", connection=_StubWRDS(), **kw)
    again = _StubWRDS()
    download("tester", connection=again, **kw)
    assert again.fundq_queries() == []        # served entirely from the cache


def test_all_null_year_does_not_stringify_a_numeric_column(tmp_path):
    """The chunking trap: ``oancfy`` is entirely NULL in 1996, so that chunk
    types it as ``object``. Concatenating on chunk dtypes silently makes the
    whole column object (and a later ``astype(str)`` renders 2022 as '2022.0').
    Dtypes must come from the Postgres declarations."""
    db = _StubWRDS(null_year=1996)
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1998,
                   intermediate_dir=tmp_path, connection=db)
    panel = pd.read_parquet(out, engine="fastparquet")
    assert panel["oancfy"].dtype == np.float64
    assert panel.loc[panel["datadate"].dt.year == 1996, "oancfy"].isna().all()
    assert panel.loc[panel["datadate"].dt.year == 1997, "oancfy"].notna().all()


def test_download_survives_missing_declarations(tmp_path):
    """No information_schema access is not fatal -- the projection still
    applies, the dtype coercion is simply skipped."""
    db = _StubWRDS(declare=False)
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1996,
                   intermediate_dir=tmp_path, connection=db, chunk_years=0)
    assert len(db.fundq_queries()) == 1        # chunk_years=0 -> single query
    assert pd.read_parquet(out, engine="fastparquet").shape[0] == 8


def test_coerce_declared_dtypes_uses_declarations_not_content():
    df = pd.DataFrame({"oancfy": pd.Series([None, None], dtype=object),
                       "gvkey": ["001000", "001001"],
                       "datadate": ["2000-03-31", "2000-06-30"]})
    decl = {"oancfy": "double precision", "gvkey": "character varying",
            "datadate": "date"}
    out = coerce_declared_dtypes(df.copy(), decl)
    assert out["oancfy"].dtype == np.float64
    assert out["gvkey"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(out["datadate"])
    assert coerce_declared_dtypes(df.copy(), {})["oancfy"].dtype == object
