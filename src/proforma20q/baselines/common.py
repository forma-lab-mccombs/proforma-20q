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
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..schema import (
    ALIASES,
    FIRM_COL,
    HORIZON_COL,
    ORIGIN_COL,
    PREDICTION_COL,
    TARGET_COL,
    normalize_columns,
)

_TARGET_RE = re.compile(r"(.+)_t(\d+)$")
_LEVEL0_RE = re.compile(r"(.+)_level_0$")


def _colnames(obj) -> list[str]:
    """Column names of a frame, or the sequence itself.

    Accepting a bare name list lets the driver decide which columns to *read*
    before paying for them -- a tabular split is ~2,547 columns and several GB.
    """
    return list(obj.columns) if hasattr(obj, "columns") else list(obj)


def feature_columns(df) -> list[str]:
    """Model input columns: the lagged levels / YoY changes and FF48 dummies.

    Excludes the target ``_t{h}`` columns and the metadata keys.
    """
    cols = []
    for c in _colnames(df):
        if _TARGET_RE.match(c):
            continue
        if "_level_" in c or "_yoy_" in c or c.startswith("indff48_"):
            cols.append(c)
    return cols


def target_columns(df, targets: list[str] | None = None) -> list[str]:
    """Sorted ``{item}_t{h}`` target columns, optionally restricted to ``targets``."""
    out = []
    for c in _colnames(df):
        m = _TARGET_RE.match(c)
        if not m:
            continue
        if targets is None or m.group(1) in targets:
            out.append(c)
    return sorted(out, key=lambda c: (_TARGET_RE.match(c).group(1), int(_TARGET_RE.match(c).group(2))))


def discover_targets(df) -> list[str]:
    """Item names that have BOTH a ``{item}_level_0`` baseline and ``{item}_t{h}``
    target columns -- the scoreable joint targets in this frame."""
    cols = _colnames(df)
    have_level0 = {m.group(1) for c in cols if (m := _LEVEL0_RE.match(c))}
    have_target = {m.group(1) for c in cols if (m := _TARGET_RE.match(c))}
    return sorted(have_level0 & have_target)


def parse_target_col(col: str) -> tuple[str, int]:
    m = _TARGET_RE.match(col)
    return m.group(1), int(m.group(2))


def forecast_block(meta: pd.DataFrame, values, target: str, horizon: int) -> pd.DataFrame:
    """One ``(target, horizon)`` block of a forecast, in the submission schema.

    The unit of assembly for a full-coverage submission. A complete forecast on
    the canonical test split is 352,962 origins x 78 targets x 20 horizons =
    **550,620,720 rows**; melting that in one call needs >20 GB before melt's own
    intermediates and raises ``ArrayMemoryError``. One block is 1/1560th of it.
    """
    firm = meta[FIRM_COL].to_numpy()
    n = len(firm)
    return pd.DataFrame({
        FIRM_COL: firm,
        TARGET_COL: np.full(n, target, dtype=object),
        ORIGIN_COL: meta[ORIGIN_COL].to_numpy(),
        HORIZON_COL: np.full(n, int(horizon), dtype=np.int64),
        PREDICTION_COL: np.asarray(values),
    })


def wide_prediction_blocks(
    meta: pd.DataFrame,
    pred_wide: pd.DataFrame,
    target_cols: list[str],
) -> Iterator[pd.DataFrame]:
    """Yield one submission-schema block per ``{item}_t{h}`` column of a wide
    prediction frame. Streaming form of :func:`wide_predictions_to_long`."""
    for col in target_cols:
        item, h = parse_target_col(col)
        yield forecast_block(meta, pred_wide[col].to_numpy(), item, h)


def wide_predictions_to_long(
    meta: pd.DataFrame,
    pred_wide: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """Reshape a wide ``{item}_t{h}`` prediction frame into the submission schema.

    ``meta`` carries ``firm`` and ``origin`` aligned row-for-row with
    ``pred_wide``. Convenience form that materializes the whole long frame --
    fine at test scale, but a full-coverage forecast is ~550M rows: stream it
    with :func:`wide_prediction_blocks` +
    :func:`~proforma20q.schema.write_forecast_blocks` instead.
    """
    blocks = list(wide_prediction_blocks(meta, pred_wide, target_cols))
    if not blocks:
        # Empty, but with the dtypes a real forecast has -- an all-object
        # sentinel would degrade `horizon` to object when concatenated.
        return pd.DataFrame({FIRM_COL: np.array([], dtype=object),
                             TARGET_COL: np.array([], dtype=object),
                             ORIGIN_COL: np.array([], dtype="datetime64[ns]"),
                             HORIZON_COL: np.array([], dtype=np.int64),
                             PREDICTION_COL: np.array([], dtype=float)})
    return pd.concat(blocks, ignore_index=True)


def tabular_columns(path) -> list[str]:
    """Column names of a tabular artifact, read from the parquet schema only."""
    import fastparquet  # noqa: PLC0415
    return [ALIASES.get(c, c) for c in fastparquet.ParquetFile(str(path)).columns]


def load_tabular(path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a tabular artifact and normalize the metadata column names.

    ``columns`` projects the read. The canonical splits are 2,547 columns and
    ~12 GB across the three of them; `naive` reads 314 of those columns (the
    four seasonal-alignment levels per item) and `fade` 1,640, so the
    projection is the difference between a baseline run that fits in memory
    and one that does not.
    """
    from ..schema import PARQUET_ENGINE
    if columns is not None:
        # translate public names back to whatever the file actually stores
        stored = set(fastparquet_columns(path))
        inverse = {v: k for k, v in ALIASES.items()}
        columns = [c if c in stored else inverse.get(c, c) for c in columns]
        columns = list(dict.fromkeys(columns))
        absent = [c for c in columns if c not in stored]
        if absent:
            # The driver derives one column list from the test split's schema and
            # applies it to all three; a train/val split missing a column would
            # otherwise degrade into a silent partial fit.
            shown = ", ".join(absent[:8])
            more = f", ... (+{len(absent) - 8} more)" if len(absent) > 8 else ""
            raise KeyError(f"{path} is missing {len(absent)} requested column(s): "
                           f"{shown}{more}")
    return normalize_columns(pd.read_parquet(path, engine=PARQUET_ENGINE,
                                             columns=columns))


def fastparquet_columns(path) -> list[str]:
    """Raw (un-normalized) column names of a parquet file."""
    import fastparquet  # noqa: PLC0415
    return list(fastparquet.ParquetFile(str(path)).columns)
