"""Cheap, fit-free (or closed-form) reference baselines.

* ``naive``  -- **seasonal random walk**: the forecast for horizon ``h`` is the
  most recent lagged level that is a whole number of years before the forecast
  quarter, i.e. ``{item}_level_{(4 - h % 4) % 4}``. Horizon 1 is predicted from
  ``level_3`` (three quarters before the origin), h=2 from ``level_2``, h=3
  from ``level_1``, h=4 from ``level_0``, then the pattern repeats (h=5 from
  ``level_3``, ...). Every base value is fiscal-quarter-aligned with the cell
  it predicts, so seasonal items are compared same-quarter-to-same-quarter.
  This is the specification behind the published leaderboard row (R^2 -0.0412
  on the Full sample); note it is *not* the plain "no change from the origin"
  random walk, whose Full-sample R^2 is ~-0.005.

* ``fade``  -- AR(1) / fade-to-mean, one ``(rho, b)`` **per (item, horizon)**
  pair (78 x 20 = 1,560 closed-form OLS fits): ``{item}_t{h} ~ {item}_level_0``
  pooled across train(+val) firm-quarters, predicting
  ``rho_{i,h} * level_0 + b_{i,h}``. Because the space is z-scored per item,
  ``rho`` fades each item toward its cross-sectional mean at its own speed --
  transient flows fade fast, cumulative stock accounts barely at all (a few
  have ``|rho| > 1``; predictions stay bounded because the inputs are the
  build-time +/-6-sigma-clipped levels). Fitting one pooled slope per horizon
  *across* items instead is a different, much weaker baseline (Full-sample
  R^2 0.106 vs 0.183) -- an earlier release shipped that by mistake.

  Pairs with fewer than ``MIN_FIT_N`` finite fit rows (never the case on a
  canonical-scale build) fall back to the bounded no-signal fade
  ``(rho=0, b=mean of the available labels)``, or to the plain random walk on
  ``level_0`` (``rho=1, b=0``) when there are no finite rows at all, so
  coverage stays finite and the common sample cannot silently shrink.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import FIRM_COL, ORIGIN_COL
from .common import discover_targets, forecast_block, parse_target_col, target_columns

#: Quarters per year -- the period of the seasonal alignment. Fixed by the
#: calendar, NOT by how many lagged level columns the build writes.
SEASONAL_PERIOD = 4

#: Number of lagged level columns the tabular view carries (``level_0..3``);
#: mirrors ``recent_levels`` in ``task.yaml``. Coincides with
#: ``SEASONAL_PERIOD`` today, which is what lets every horizon find a
#: same-fiscal-quarter base among the built lags.
N_LEVEL_LAGS = 4

#: Minimum finite fit rows for a per-(item, horizon) fade fit; below it the
#: pair falls back to the bounded no-signal fade. Matches the internal
#: pipeline's threshold. Never binding at canonical scale, so it is a module
#: constant rather than a public knob.
MIN_FIT_N = 100


def seasonal_lag(horizon: int) -> int:
    """Lag of the level column the seasonal random walk predicts ``horizon`` from.

    ``(4 - h % 4) % 4``: the youngest available lag whose distance to the
    forecast quarter, ``h + lag``, is a whole number of years.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return (SEASONAL_PERIOD - horizon % SEASONAL_PERIOD) % SEASONAL_PERIOD


def _seasonal_base(df: pd.DataFrame, item: str, horizon: int) -> np.ndarray:
    col = f"{item}_level_{seasonal_lag(horizon)}"
    if col not in df.columns:
        raise KeyError(
            f"naive needs {col} to predict {item}_t{horizon} (seasonal random "
            f"walk reads level_0..{N_LEVEL_LAGS - 1}); it is absent from the "
            "frame -- either projected out of the read, or never built "
            "(task.yaml recent_levels)")
    return df[col].to_numpy()


def iter_naive_blocks(test_df: pd.DataFrame, targets: list[str] | None = None,
                      target_cols: list[str] | None = None):
    """Yield the seasonal-random-walk forecast one ``(target, horizon)`` block
    at a time.

    ``target_cols`` names the ``{item}_t{h}`` cells to forecast. The random walk
    reads none of them -- it only needs their names -- so the caller can supply
    them from the parquet schema and never load 1,560 columns of truth.
    """
    targets = targets or discover_targets(test_df)
    if target_cols is None:
        target_cols = target_columns(test_df, targets)
    for col in target_cols:
        item, h = parse_target_col(col)
        yield forecast_block(test_df, _seasonal_base(test_df, item, h), item, h)


def predict_naive(test_df: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Seasonal random walk: prediction = ``{item}_level_{(4 - h % 4) % 4}``.

    Materializes the whole forecast; see :func:`iter_naive_blocks` for the
    streaming form the driver uses at benchmark scale.
    """
    return pd.concat(iter_naive_blocks(test_df, targets), ignore_index=True)


def _fit_fade(fit_dfs, targets: list[str], horizons: list[int],
              min_fit_n: int = MIN_FIT_N) -> dict[tuple[str, int], tuple[float, float]]:
    """AR(1) per ``(item, horizon)``: 1,560 closed-form OLS fits at full scale.

    For each pair, fits ``y = rho * x + b`` with ``x = {item}_level_0``,
    ``y = {item}_t{h}``, pooled over every training row with a finite pair.

    ``fit_dfs`` is the list of frames to pool (train, then val). They are
    stacked per (target, frame) rather than concatenated up front: the pooled
    arrays are identical, but concatenating two 2,547-column splits first costs
    ~8 GB on the canonical build for columns the fit never reads.

    Fallbacks (all finite, so a fade forecast can never shrink a common
    sample): fewer than ``min_fit_n`` finite rows -> ``(0, label mean)``; no
    finite rows at all -> ``(1, 0)``, the *plain* random walk on ``level_0``
    (not the seasonal walk ``naive`` uses); ``x`` constant in the fit sample
    (e.g. an all-zero item at long horizons) -> ``(0, label mean)``.
    """
    if isinstance(fit_dfs, pd.DataFrame):
        fit_dfs = [fit_dfs]
    coeffs: dict[tuple[str, int], tuple[float, float]] = {}
    level0 = [{t: df[f"{t}_level_0"].to_numpy(np.float64) for t in targets
               if f"{t}_level_0" in df.columns} for df in fit_dfs]
    for t in targets:
        for h in horizons:
            ycol = f"{t}_t{h}"
            xv, yv = [], []
            for df, lv0 in zip(fit_dfs, level0):
                if t not in lv0 or ycol not in df.columns:
                    continue
                x = lv0[t]
                y = df[ycol].to_numpy(np.float64)
                m = np.isfinite(x) & np.isfinite(y)
                if m.any():
                    xv.append(x[m])
                    yv.append(y[m])
            if not xv:
                # no data at all -> plain random walk on level_0
                coeffs[(t, h)] = (1.0, 0.0)
                continue
            x = np.concatenate(xv) if len(xv) > 1 else xv[0]
            y = np.concatenate(yv) if len(yv) > 1 else yv[0]
            if len(x) < min_fit_n:
                coeffs[(t, h)] = (0.0, float(y.mean()))
                continue
            xbar, ybar = x.mean(), y.mean()
            sxx = float(((x - xbar) ** 2).sum())
            if sxx <= 1e-12:
                coeffs[(t, h)] = (0.0, float(ybar))
            else:
                rho = float(((x - xbar) * (y - ybar)).sum() / sxx)
                b = float(ybar - rho * xbar)
                coeffs[(t, h)] = (rho, b)
    return coeffs


def iter_fade_blocks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    val_df: pd.DataFrame | None = None,
    target_cols: list[str] | None = None,
):
    """Fit the per-(item, horizon) AR(1) on train(+val), then yield one block
    per ``(target, horizon)``."""
    targets = targets or discover_targets(test_df)
    tcols = target_columns(test_df, targets) if target_cols is None else target_cols
    horizons = sorted({parse_target_col(c)[1] for c in tcols})

    coeffs = _fit_fade([train_df] if val_df is None else [train_df, val_df],
                       targets, horizons)

    for col in tcols:
        item, h = parse_target_col(col)
        rho, b = coeffs.get((item, h), (1.0, 0.0))
        yield forecast_block(
            test_df, rho * test_df[f"{item}_level_0"].to_numpy(np.float64) + b, item, h)


def predict_fade(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    val_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit the per-(item, horizon) AR(1) on train(+val) and predict on ``test_df``."""
    return pd.concat(iter_fade_blocks(train_df, test_df, targets, val_df),
                     ignore_index=True)
