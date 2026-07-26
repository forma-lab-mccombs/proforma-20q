"""Cheap, fit-free (or single-coefficient) reference baselines.

* ``naive``  -- random walk: the forecast at every horizon is the regularized
  value at the origin, ``{item}_level_0``. In change space this predicts *zero
  change*, so it is the natural R^2 anchor (its change-space MSE is exactly the
  realized-change variance, and its R^2 is ~0 by construction on the pooled
  denominator). Every model worth shipping must beat it.

* ``fade``  -- a pooled AR(1) / fade-to-mean. For each horizon h it fits ONE
  slope + intercept by pooling ``{item}_t{h} ~ {item}_level_0`` across every
  target item and every train(+val) firm-quarter, then predicts
  ``rho_h * level_0 + b_h``. Because the space is z-scored (item mean ~ 0),
  ``rho_h in (0, 1)`` fades the level toward the cross-sectional mean as the
  horizon grows -- a stronger, still-trivial anchor than the pure random walk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import FIRM_COL, ORIGIN_COL
from .common import discover_targets, forecast_block, parse_target_col, target_columns


def iter_naive_blocks(test_df: pd.DataFrame, targets: list[str] | None = None,
                      target_cols: list[str] | None = None):
    """Yield the random-walk forecast one ``(target, horizon)`` block at a time.

    ``target_cols`` names the ``{item}_t{h}`` cells to forecast. The random walk
    reads none of them -- it only needs their names -- so the caller can supply
    them from the parquet schema and never load 1,560 columns of truth.
    """
    targets = targets or discover_targets(test_df)
    if target_cols is None:
        target_cols = target_columns(test_df, targets)
    for col in target_cols:
        item, h = parse_target_col(col)
        yield forecast_block(test_df, test_df[f"{item}_level_0"].to_numpy(), item, h)


def predict_naive(test_df: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Random-walk forecast: prediction = ``{item}_level_0`` for every horizon.

    Materializes the whole forecast; see :func:`iter_naive_blocks` for the
    streaming form the driver uses at benchmark scale.
    """
    return pd.concat(iter_naive_blocks(test_df, targets), ignore_index=True)


def _fit_fade(fit_dfs, targets: list[str], horizons: list[int]) -> dict[int, tuple[float, float]]:
    """Pooled AR(1): one (rho, b) per horizon, pooling all items and rows.

    Fits ``y = rho * x + b`` where ``x = {item}_level_0``, ``y = {item}_t{h}``,
    stacked over every item and every training row (finite pairs only).

    ``fit_dfs`` is the list of frames to pool (train, then val). They are stacked
    per (target, frame) rather than concatenated up front: the pooled arrays are
    identical, but concatenating two 2,547-column splits first costs ~8 GB on the
    canonical build for columns the fit never reads.
    """
    if isinstance(fit_dfs, pd.DataFrame):
        fit_dfs = [fit_dfs]
    coeffs: dict[int, tuple[float, float]] = {}
    level0 = [{t: df[f"{t}_level_0"].to_numpy(np.float64) for t in targets
               if f"{t}_level_0" in df.columns} for df in fit_dfs]
    for h in horizons:
        xs, ys = [], []
        for t in targets:
            for df, lv0 in zip(fit_dfs, level0):
                ycol = f"{t}_t{h}"
                if ycol not in df.columns or t not in lv0:
                    continue
                x = lv0[t]
                y = df[ycol].to_numpy(np.float64)
                m = np.isfinite(x) & np.isfinite(y)
                if m.any():
                    xs.append(x[m])
                    ys.append(y[m])
        if not xs:
            coeffs[h] = (1.0, 0.0)  # degenerate -> random walk
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        # OLS slope/intercept via closed form.
        xbar, ybar = x.mean(), y.mean()
        sxx = float(((x - xbar) ** 2).sum())
        if sxx <= 1e-12:
            coeffs[h] = (0.0, ybar)
        else:
            rho = float(((x - xbar) * (y - ybar)).sum() / sxx)
            b = float(ybar - rho * xbar)
            coeffs[h] = (rho, b)
    return coeffs


def iter_fade_blocks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    val_df: pd.DataFrame | None = None,
    target_cols: list[str] | None = None,
):
    """Fit the pooled AR(1) on train(+val), then yield one block per
    ``(target, horizon)``."""
    targets = targets or discover_targets(test_df)
    tcols = target_columns(test_df, targets) if target_cols is None else target_cols
    horizons = sorted({parse_target_col(c)[1] for c in tcols})

    coeffs = _fit_fade([train_df] if val_df is None else [train_df, val_df],
                       targets, horizons)

    for col in tcols:
        item, h = parse_target_col(col)
        rho, b = coeffs.get(h, (1.0, 0.0))
        yield forecast_block(
            test_df, rho * test_df[f"{item}_level_0"].to_numpy(np.float64) + b, item, h)


def predict_fade(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    val_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit the pooled AR(1) on train(+val) and predict on ``test_df``."""
    return pd.concat(iter_fade_blocks(train_df, test_df, targets, val_df),
                     ignore_index=True)
