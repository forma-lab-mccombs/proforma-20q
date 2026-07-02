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
from .naive import predict_fade, predict_naive
from .sklearn_models import fit_predict_sklearn

BASELINES = ("naive", "fade", "elasticnet", "linear")


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

__all__ = ["BASELINES", "run_baseline", "run_baselines"]
