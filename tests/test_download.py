"""Unit tests for the WRDS-free parts of download (the CCM link merge).

No WRDS connection: attach_permno is pure pandas over a Compustat panel and a
CCM link table, so the port's permno-attachment (date-range imposition, NaN link
handling, column rename/drop) is fully testable on synthetic frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proforma20q.download import attach_permno


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
