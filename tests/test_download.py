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
import pytest

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
        return self._as_wrds_dtypes(pd.DataFrame(recs, columns=cols))

    @staticmethod
    def _as_wrds_dtypes(df):
        """`wrds.raw_sql` defaults to dtype_backend="numpy_nullable", so text
        columns arrive as `string[python]`, not `object`. A chunk read back from
        the parquet cache arrives as `object` -- the mismatch that makes a
        resumed pull die in merge_asof."""
        for c in ("gvkey", "tic", "conm", "indfmt", "datafmt", "consol", "popsrc"):
            if c in df.columns:
                df[c] = df[c].astype("string")
        return df

    def raw_sql(self, sql):
        self.queries.append(sql)
        if "information_schema" in sql:
            if not self.declare:
                raise RuntimeError("no information_schema access")
            return self._declarations()
        if "comp.fundq" in sql:
            return self._fundq(sql)
        if "co_industry" in sql:
            return self._as_wrds_dtypes(pd.DataFrame({
                "datadate": pd.to_datetime(["1969-12-31", "1969-12-31"]),
                "gvkey": ["001000", "001001"],
                "sich": [3500.0, 2000.0], "naicsh": [334111.0, 311111.0]}))
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
    assert len(cached) == 3
    assert [c.split("__")[0] for c in cached] == ["fundq_1996_1996", "fundq_1997_1997",
                                                  "fundq_1998_1998"]


def test_download_reuses_cached_chunks(tmp_path):
    kw = dict(out_dir=tmp_path / "raw", start_year=1996, end_year=1997,
              intermediate_dir=tmp_path)
    first = download("tester", connection=_StubWRDS(), **kw)
    fresh = pd.read_parquet(first, engine="fastparquet")
    again = _StubWRDS()
    second = download("tester", connection=again, **kw)
    assert again.fundq_queries() == []        # served entirely from the cache
    # ... and the resumed panel is the same panel, not just the same row count.
    pd.testing.assert_frame_equal(pd.read_parquet(second, engine="fastparquet"), fresh)


def test_resumed_pull_survives_the_wrds_nullable_string_dtypes(tmp_path):
    """A cached chunk comes back from parquet as `object`; `co_industry` is
    always pulled fresh and comes back as `string[python]`. Without normalizing
    the join key, merge_asof raises -- after the Duo prompt and after re-pulling
    every uncached year. Resume is the whole point of chunking, so this path
    must work."""
    kw = dict(out_dir=tmp_path / "raw", intermediate_dir=tmp_path)
    download("tester", connection=_StubWRDS(), start_year=1996, end_year=1997, **kw)
    resumed = _StubWRDS()
    out = download("tester", connection=resumed, start_year=1996, end_year=1998, **kw)
    assert len(resumed.fundq_queries()) == 1        # 1996-97 cached, 1998 fresh
    panel = pd.read_parquet(out, engine="fastparquet")
    assert len(panel) == 24
    assert panel["sich"].notna().all()              # the merge actually matched


def test_cached_chunk_is_not_reused_under_a_different_projection(tmp_path):
    """A chunk pulled with another column set is not interchangeable: reusing it
    by year alone unions two schemas, leaving a column populated for some years
    and silently NaN for others."""
    kw = dict(out_dir=tmp_path / "raw", intermediate_dir=tmp_path,
              start_year=1996, end_year=1997)
    narrow = ["gvkey", "datadate", "fyearq", "fqtr", "indfmt", "datafmt",
              "consol", "popsrc", "atq"]
    download("tester", connection=_StubWRDS(), columns=narrow, **kw)
    db = _StubWRDS()
    out = download("tester", connection=db, **kw)          # default 82 columns
    assert len(db.fundq_queries()) == 2                    # re-pulled, not reused
    panel = pd.read_parquet(out, engine="fastparquet")
    assert "ltq" in panel.columns and panel["ltq"].notna().all()
    assert len(list((tmp_path / "raw_chunks").glob("fundq_1996_1996__*.parquet"))) == 2


def test_all_null_year_does_not_stringify_a_numeric_column(tmp_path):
    """The chunking trap: ``oancfy`` is entirely NULL in 1996, so that chunk
    types it as ``object``. Concatenating on chunk dtypes silently makes the
    whole column object (and a later ``astype(str)`` renders 2022 as '2022.0').
    Dtypes must come from the Postgres declarations -- asserted on the CACHED
    CHUNK, since a coercion applied only after the concat would pass a check on
    the final panel alone."""
    db = _StubWRDS(null_year=1996)
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1998,
                   intermediate_dir=tmp_path, connection=db)
    chunk = next((tmp_path / "raw_chunks").glob("fundq_1996_1996__*.parquet"))
    assert pd.read_parquet(chunk, engine="fastparquet")["oancfy"].dtype == np.float64
    panel = pd.read_parquet(out, engine="fastparquet")
    assert panel["oancfy"].dtype == np.float64
    assert panel.loc[panel["datadate"].dt.year == 1996, "oancfy"].isna().all()
    assert panel.loc[panel["datadate"].dt.year == 1997, "oancfy"].notna().all()


def test_missing_declarations_still_repairs_the_all_null_chunk(tmp_path):
    """Without information_schema the dtype source is gone but chunking is still
    on by default -- i.e. exactly the trap, on the default path. Fall back to
    cross-chunk inference rather than silently doing nothing."""
    db = _StubWRDS(declare=False, null_year=1996)
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1998,
                   intermediate_dir=tmp_path, connection=db)
    panel = pd.read_parquet(out, engine="fastparquet")
    assert panel["oancfy"].dtype == np.float64
    assert panel.loc[panel["datadate"].dt.year == 1997, "oancfy"].notna().all()


def test_download_survives_missing_declarations(tmp_path):
    """No information_schema access is not fatal -- the projection still
    applies, the dtype coercion is simply skipped."""
    db = _StubWRDS(declare=False)
    out = download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1996,
                   intermediate_dir=tmp_path, connection=db, chunk_years=0)
    assert len(db.fundq_queries()) == 1        # chunk_years=0 -> single query
    assert pd.read_parquet(out, engine="fastparquet").shape[0] == 8


def test_projected_column_absent_from_fundq_is_fatal(tmp_path):
    """Silently dropping a projected column deletes a benchmark feature: a
    `{base}q` item exists only via its `{base}y` source, so a missing source
    shrinks the feature set and changes the artifacts' shape, with an md5
    mismatch as the only signal."""
    db = _StubWRDS()
    with pytest.raises(KeyError, match="absent from comp.fundq"):
        download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1996,
                 intermediate_dir=tmp_path, connection=db,
                 columns=["gvkey", "datadate", "no_such_column"])


@pytest.mark.parametrize("kwargs", [
    dict(start_year=2000, end_year=1990),      # transposed years
    dict(start_year=1996, end_year=1996, chunk_years=-1),   # infinite loop
    dict(start_year=1996, end_year=1996, columns=["gvkey", "atq FROM comp.funda; --"]),
])
def test_bad_arguments_fail_before_the_connection_is_used(tmp_path, kwargs):
    """Argument validation must happen before any query: a typo should not cost
    a Duo round trip, and `chunk_years=-1` used to hang forever after auth."""
    db = _StubWRDS()
    with pytest.raises((ValueError, KeyError)):
        download("tester", out_dir=tmp_path / "raw", intermediate_dir=tmp_path,
                 connection=db, **kwargs)
    assert db.queries == []


def test_year_chunk_windows():
    from proforma20q.download import _year_chunks

    assert list(_year_chunks(1996, 1998, 1)) == [(1996, 1996), (1997, 1997), (1998, 1998)]
    assert list(_year_chunks(1996, 1996, 1)) == [(1996, 1996)]
    assert list(_year_chunks(1996, 2000, 2)) == [(1996, 1997), (1998, 1999), (2000, 2000)]
    assert list(_year_chunks(1996, 1998, 99)) == [(1996, 1998)]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            list(_year_chunks(1996, 1998, bad))
    with pytest.raises(ValueError):
        list(_year_chunks(2000, 1990, 1))


def test_coerce_declared_dtypes_uses_declarations_not_content():
    df = pd.DataFrame({"oancfy": pd.Series([None, None], dtype=object),
                       "gvkey": ["001000", "001001"],
                       "datadate": ["2000-03-31", "2000-06-30"]})
    decl = {"oancfy": "double precision", "gvkey": "character varying",
            "datadate": "date"}
    out = coerce_declared_dtypes(df.copy(), decl)
    assert out["oancfy"].dtype == np.float64
    assert pd.api.types.is_datetime64_any_dtype(out["datadate"])
    # A declared-varchar column is left alone. Asserted as "still text, values
    # intact" rather than "dtype is object": pandas 3 infers a string column as
    # StringDtype, and which one pandas picked is not what this function decides.
    assert not pd.api.types.is_numeric_dtype(out["gvkey"])
    assert not pd.api.types.is_datetime64_any_dtype(out["gvkey"])
    assert list(out["gvkey"]) == ["001000", "001001"]
    # no declarations -> nothing is coerced, so the all-NULL column stays as it was
    assert coerce_declared_dtypes(df.copy(), {})["oancfy"].dtype == df["oancfy"].dtype


def test_crsp_link_config_is_validated_like_the_projection():
    """Every field here is interpolated into SQL from the same trust level as the
    fundq projection, so it gets the same treatment."""
    from unittest.mock import patch

    import proforma20q.download as dl

    good = dl.crsp_link_config()
    assert good["table"] == "crsp.ccmxpf_lnkhist"
    assert good["impose_link_dates"] is True

    for bad in ({"table": "crsp.lnk; DROP TABLE x"},
                {"linktype": ["LU", "L'U"]},
                {"linkprim": ["P", "C; --"]}):
        cfg = {"universe": {"crsp_link": {**good, **bad}}}
        with patch.object(dl, "load_task_config", lambda: cfg):
            with pytest.raises(ValueError):
                dl.crsp_link_config()


def test_chunk_query_is_ordered(tmp_path):
    """Postgres row order is unspecified without ORDER BY, and the input order
    decides which of a duplicate firm-quarter pair survives de-duplication
    downstream (pandas multi-column sort_values is a stable lexsort). Without
    this the 'stable sort' determinism fix is conditional on the DB."""
    db = _StubWRDS()
    download("tester", out_dir=tmp_path / "raw", start_year=1996, end_year=1997,
             intermediate_dir=tmp_path, connection=db)
    for q in db.fundq_queries():
        assert "ORDER BY f.gvkey, f.datadate" in q


def test_impose_link_dates_is_wired_to_the_config(tmp_path):
    """task.yaml is the declared source of truth for the sample definition, so a
    key in it that changes nothing is worse than no key at all."""
    from unittest.mock import patch

    import proforma20q.download as dl

    comp = pd.DataFrame({"gvkey": ["001", "001"],
                         "datadate": pd.to_datetime(["2000-03-31", "2010-03-31"]),
                         "atq": [1.0, 2.0]})
    link = pd.DataFrame([_link_row("001", 10001, 20001, "1999-01-01", "2005-12-31")])

    assert len(dl.attach_permno(comp, link)) == 1                          # filtered
    assert len(dl.attach_permno(comp, link, impose_link_dates=False)) == 2  # not

    cfg = load_task_config_copy = dl.load_task_config()
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["universe"]["crsp_link"]["impose_link_dates"] = False
    with patch.object(dl, "load_task_config", lambda: cfg):
        assert dl.crsp_link_config()["impose_link_dates"] is False
