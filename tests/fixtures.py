"""Synthetic fixtures for the test suite. No real data anywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_raw(n_firms: int = 24, start="1996Q1", end="2013Q4", seed: int = 7) -> pd.DataFrame:
    """A tiny Compustat-like raw panel (post-``compustat_with_permno``) with a
    handful of pf_full items, YTD cash-flow sources, and industry codes."""
    rng = np.random.default_rng(seed)
    firms = [f"{i:05d}" for i in range(n_firms)]
    quarters = pd.period_range(start, end, freq="Q")
    rows = []
    for f in firms:
        base = rng.lognormal(4, 1)
        oancf_ytd = 0.0
        for q in quarters:
            fqtr = q.quarter
            if fqtr == 1:
                oancf_ytd = 0.0
            rev = base * rng.lognormal(0, 0.15) * (1 + 0.01 * (q.year - 1996))
            cogs = rev * rng.uniform(0.5, 0.8)
            ni = rev - cogs - rev * rng.uniform(0.05, 0.2)
            oancf_ytd += ni * rng.uniform(0.7, 1.3)
            rows.append(dict(
                gvkey=f, datadate=q.to_timestamp(how="end").normalize(),
                fyearq=q.year, fqtr=fqtr,
                sich=float(rng.choice([2000, 3500, 7370, 1200])),
                naicsh=float(rng.choice([311111, 334111])),
                revtq=rev, cogsq=cogs, niq=ni,
                atq=base * rng.uniform(2, 4), ltq=base * rng.uniform(1, 2),
                seqq=base * rng.uniform(0.5, 1.5),
                oancfy=oancf_ytd, capxy=(base * 0.05 * fqtr) * rng.uniform(0.8, 1.2),
            ))
    return pd.DataFrame(rows)


def synthetic_truth(n_firms: int = 30, n_q: int = 6, targets=("niq", "revtq", "atq"),
                    horizons=range(1, 6), seed: int = 0) -> pd.DataFrame:
    """A wide truth frame in the eval schema (``firm``/``origin`` +
    ``{t}_level_0`` + ``{t}_t{h}``), already in a z-scored-like space."""
    rng = np.random.default_rng(seed)
    firms = [f"{i:06d}" for i in range(n_firms)]
    origins = pd.period_range("2010Q1", periods=n_q, freq="Q").to_timestamp(how="end")
    rows = [{"firm": f, "origin": o} for f in firms for o in origins]
    truth = pd.DataFrame(rows)
    for t in targets:
        truth[f"{t}_level_0"] = rng.standard_normal(len(truth))
        for h in horizons:
            truth[f"{t}_t{h}"] = (truth[f"{t}_level_0"] + 0.3 * h
                                  + rng.standard_normal(len(truth)) * 0.5)
    return truth


def forecast_from_truth(truth: pd.DataFrame, noise: float, sigma: float | None = None,
                        seed: int = 1) -> pd.DataFrame:
    """A long forecast: each target's future value + Gaussian noise."""
    rng = np.random.default_rng(seed)
    targets = sorted({c.rsplit("_t", 1)[0] for c in truth.columns
                      if "_t" in c and c.rsplit("_t", 1)[-1].isdigit()})
    horizons = sorted({int(c.rsplit("_t", 1)[-1]) for c in truth.columns
                       if "_t" in c and c.rsplit("_t", 1)[-1].isdigit()})
    recs = []
    for _, r in truth.iterrows():
        for t in targets:
            for h in horizons:
                rec = {"firm": r["firm"], "target": t, "origin": r["origin"],
                       "horizon": h, "prediction": r[f"{t}_t{h}"] + rng.standard_normal() * noise}
                if sigma is not None:
                    rec["sigma"] = sigma
                recs.append(rec)
    return pd.DataFrame(recs)
