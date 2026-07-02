"""Shared helpers for the reference baselines.

Every baseline consumes the *tabular* build artifacts (``tabular_{split}``) and
emits a forecast frame in the submission schema. The tabular frame is wide:

* metadata: ``firm`` / ``origin`` (aka ``firm_id`` / ``quarter``)
* features: ``{item}_level_0..3``, ``{item}_yoy_0..7``, ``scale_level_0``,
  ``indff48_0..47``
* targets:  ``{item}_t1..t20``   (regularized future values)
* baseline: ``{item}_level_0``   (regularized value at origin t)

all in the regularized target space.
"""
from __future__ import annotations

import re

import pandas as pd

from ..schema import (
    FIRM_COL,
    HORIZON_COL,
    ORIGIN_COL,
    PREDICTION_COL,
    TARGET_COL,
    normalize_columns,
)

_TARGET_RE = re.compile(r"(.+)_t(\d+)$")
_LEVEL0_RE = re.compile(r"(.+)_level_0$")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: the lagged levels / YoY changes and FF48 dummies.

    Excludes the target ``_t{h}`` columns and the metadata keys.
    """
    cols = []
    for c in df.columns:
        if _TARGET_RE.match(c):
            continue
        if "_level_" in c or "_yoy_" in c or c.startswith("indff48_"):
            cols.append(c)
    return cols


def target_columns(df: pd.DataFrame, targets: list[str] | None = None) -> list[str]:
    """Sorted ``{item}_t{h}`` target columns, optionally restricted to ``targets``."""
    out = []
    for c in df.columns:
        m = _TARGET_RE.match(c)
        if not m:
            continue
        if targets is None or m.group(1) in targets:
            out.append(c)
    return sorted(out, key=lambda c: (_TARGET_RE.match(c).group(1), int(_TARGET_RE.match(c).group(2))))


def discover_targets(df: pd.DataFrame) -> list[str]:
    """Item names that have BOTH a ``{item}_level_0`` baseline and ``{item}_t{h}``
    target columns -- the scoreable joint targets in this frame."""
    have_level0 = {m.group(1) for c in df.columns if (m := _LEVEL0_RE.match(c))}
    have_target = {m.group(1) for c in df.columns if (m := _TARGET_RE.match(c))}
    return sorted(have_level0 & have_target)


def parse_target_col(col: str) -> tuple[str, int]:
    m = _TARGET_RE.match(col)
    return m.group(1), int(m.group(2))


def wide_predictions_to_long(
    meta: pd.DataFrame,
    pred_wide: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """Melt a wide ``{item}_t{h}`` prediction frame into the submission schema.

    ``meta`` carries ``firm`` and ``origin`` aligned row-for-row with
    ``pred_wide``.
    """
    frame = pred_wide[target_cols].copy()
    frame[FIRM_COL] = meta[FIRM_COL].to_numpy()
    frame[ORIGIN_COL] = meta[ORIGIN_COL].to_numpy()
    long = frame.melt(
        id_vars=[FIRM_COL, ORIGIN_COL],
        value_vars=target_cols,
        var_name="_tcol",
        value_name=PREDICTION_COL,
    )
    split = long["_tcol"].str.rsplit("_t", n=1, expand=True)
    long[TARGET_COL] = split[0]
    long[HORIZON_COL] = split[1].astype(int)
    return long[[FIRM_COL, TARGET_COL, ORIGIN_COL, HORIZON_COL, PREDICTION_COL]]


def load_tabular(path) -> pd.DataFrame:
    """Read a tabular artifact and normalize the metadata column names."""
    from ..schema import PARQUET_ENGINE
    return normalize_columns(pd.read_parquet(path, engine=PARQUET_ENGINE))
