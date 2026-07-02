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
from .common import discover_targets, parse_target_col, target_columns, wide_predictions_to_long


def predict_naive(test_df: pd.DataFrame, targets: list[str] | None = None) -> pd.DataFrame:
    """Random-walk forecast: prediction = ``{item}_level_0`` for every horizon."""
    targets = targets or discover_targets(test_df)
    tcols = target_columns(test_df, targets)
    pred = pd.DataFrame(
        {col: test_df[f"{parse_target_col(col)[0]}_level_0"].to_numpy() for col in tcols},
        index=test_df.index,
    )
    return wide_predictions_to_long(test_df, pred, tcols)


def _fit_fade(train_df: pd.DataFrame, targets: list[str], horizons: list[int]) -> dict[int, tuple[float, float]]:
    """Pooled AR(1): one (rho, b) per horizon, pooling all items and rows.

    Fits ``y = rho * x + b`` where ``x = {item}_level_0``, ``y = {item}_t{h}``,
    stacked over every item and every training row (finite pairs only).
    """
    coeffs: dict[int, tuple[float, float]] = {}
    level0 = {t: train_df[f"{t}_level_0"].to_numpy(np.float64) for t in targets
              if f"{t}_level_0" in train_df.columns}
    for h in horizons:
        xs, ys = [], []
        for t in targets:
            ycol = f"{t}_t{h}"
            if ycol not in train_df.columns or t not in level0:
                continue
            x = level0[t]
            y = train_df[ycol].to_numpy(np.float64)
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


def predict_fade(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list[str] | None = None,
    val_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit the pooled AR(1) on train(+val) and predict on ``test_df``."""
    targets = targets or discover_targets(test_df)
    tcols = target_columns(test_df, targets)
    horizons = sorted({parse_target_col(c)[1] for c in tcols})

    fit_df = train_df if val_df is None else pd.concat([train_df, val_df], axis=0, ignore_index=True)
    coeffs = _fit_fade(fit_df, targets, horizons)

    cols = {}
    for col in tcols:
        item, h = parse_target_col(col)
        rho, b = coeffs.get(h, (1.0, 0.0))
        cols[col] = rho * test_df[f"{item}_level_0"].to_numpy(np.float64) + b
    pred = pd.DataFrame(cols, index=test_df.index)
    return wide_predictions_to_long(test_df, pred, tcols)
