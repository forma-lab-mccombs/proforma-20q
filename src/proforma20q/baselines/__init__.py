"""Reference baselines for ProForma-20Q.

Cheap to run, they let a user confirm their pipeline reproduces the published
baseline numbers before scoring their own model. They are NOT the Forma model.

* ``naive``      random walk (change-space zero anchor)
* ``fade``       pooled AR(1) / fade-to-mean
* ``elasticnet`` per-horizon CV'd ElasticNet, shared across targets
* ``linear``     plain OLS

Each returns a forecast in the submission schema. Use :func:`run_baseline` for a
single one or :func:`run_baselines` to produce a whole suite of forecast files.
"""
from __future__ import annotations

import pandas as pd

from .common import discover_targets
from .naive import iter_fade_blocks, iter_naive_blocks, predict_fade, predict_naive
from .sklearn_models import fit_predict_sklearn, iter_sklearn_blocks

BASELINES = ("naive", "fade", "elasticnet", "linear")


def iter_baseline_blocks(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    target_cols: list[str] | None = None,
    verbose: bool = True,
):
    """Yield ``name``'s forecast one ``(target, horizon)`` block at a time.

    The streaming form of :func:`run_baseline`. A full-coverage forecast on the
    canonical test split is ~550M rows / several GB; assembling it as one frame
    OOMs, so the driver pairs this with
    :func:`~proforma20q.schema.write_forecast_blocks`.

    ``target_cols`` overrides which ``{item}_t{h}`` cells to forecast (default:
    whatever the frames carry), so a caller that projected the truth columns out
    of its read can still say what to predict.
    """
    targets = targets or discover_targets(test_df)
    if not targets:
        # `discover_targets` needs both `{item}_level_0` and `{item}_t{h}` in the
        # frame. On a column-projected read the truth columns are gone by design,
        # so silently yielding nothing is the failure mode to prevent.
        raise ValueError(
            "no scoreable targets: pass `targets=` explicitly (and `target_cols=` "
            "if the frame was read with a column projection)")
    if name == "naive":
        return iter_naive_blocks(test_df, targets=targets, target_cols=target_cols)
    if name == "fade":
        return iter_fade_blocks(train_df, test_df, targets=targets, val_df=val_df,
                                target_cols=target_cols)
    if name in ("elasticnet", "linear"):
        return iter_sklearn_blocks(name, train_df, test_df, val_df=val_df,
                                   targets=targets, target_cols=target_cols,
                                   verbose=verbose)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")


def run_baseline(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit ``name`` and return its forecast frame (submission schema)."""
    targets = targets or discover_targets(test_df)
    if name == "naive":
        return predict_naive(test_df, targets=targets)
    if name == "fade":
        return predict_fade(train_df, test_df, targets=targets, val_df=val_df)
    if name in ("elasticnet", "linear"):
        return fit_predict_sklearn(name, train_df, test_df, val_df=val_df,
                                   targets=targets, verbose=verbose)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")


from .run import run_baselines  # noqa: E402  (re-export; imports this module)

__all__ = ["BASELINES", "iter_baseline_blocks", "run_baseline", "run_baselines"]
