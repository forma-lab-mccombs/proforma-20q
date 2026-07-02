"""Linear-family reference baselines on the tabular feature matrix.

Ports the Forma research repo's ``SklearnBaselineModel`` semantics:

* One model per ``(target, horizon)`` on the shared feature matrix (lagged
  regularized levels, YoY changes, FF48 dummies).
* **elasticnet** -- a per-horizon ``(alpha, l1_ratio)`` grid searched by
  cross-validation on a single reference target (``niq``) over the 2002-2009
  validation fold, then REUSED for that horizon across all 78 targets (the grid
  is searched H times, not |targets|*H). The final model refits on train+val.
* **linear** -- plain OLS; no hyperparameters; refits on train+val.

Rows with any NaN feature or NaN target are dropped before fitting/predicting.
Everything runs in the regularized target space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from .common import (
    discover_targets,
    feature_columns,
    parse_target_col,
    target_columns,
    wide_predictions_to_long,
)

# Canonical R13 ElasticNet grid (configs/r13_baselines.yaml).
ELASTICNET_GRID = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
}
ELASTICNET_MAX_ITER = 10000
HPARAM_REFERENCE_TARGET = "niq"


def _make_estimator(model_type: str, **params):
    if model_type == "linear":
        return LinearRegression(**params)
    if model_type == "elasticnet":
        return ElasticNet(max_iter=ELASTICNET_MAX_ITER, **params)
    raise ValueError(f"unknown linear-family baseline {model_type!r}")


def _xy(df: pd.DataFrame, feat_cols: list[str], target_col: str):
    """Feature/target matrices for one horizon with NaN rows dropped."""
    X = df[feat_cols]
    y = df[target_col]
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask].astype("float32"), y[mask].astype("float32")


def _search_hparams(Xtr, ytr, Xval, yval, grid) -> dict:
    """Grid search over a PredefinedSplit validation fold; return best params."""
    Xc = pd.concat([Xtr, Xval], axis=0)
    yc = pd.concat([ytr, yval], axis=0)
    test_fold = np.concatenate([np.full(len(Xtr), -1), np.full(len(Xval), 0)])
    gs = GridSearchCV(
        estimator=_make_estimator("elasticnet"),
        param_grid=grid,
        cv=PredefinedSplit(test_fold),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=False,
    )
    gs.fit(Xc, yc)
    return gs.best_params_


def fit_predict_sklearn(
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit a linear-family baseline and return its forecast (submission schema)."""
    targets = targets or discover_targets(test_df)
    feat_cols = feature_columns(train_df)
    tcols = target_columns(train_df, targets)
    horizons = sorted({parse_target_col(c)[1] for c in tcols})
    log = print if verbose else (lambda *a, **k: None)

    # --- ElasticNet: shared per-horizon hyperparameters (searched on niq) ---
    params_by_h: dict[int, dict] = {}
    if model_type == "elasticnet" and val_df is not None:
        ref = HPARAM_REFERENCE_TARGET
        for h in horizons:
            ref_col = f"{ref}_t{h}"
            if ref_col not in train_df.columns:
                continue
            Xtr, ytr = _xy(train_df, feat_cols, ref_col)
            Xval, yval = _xy(val_df, feat_cols, ref_col)
            if len(Xval) == 0 or len(Xtr) == 0:
                continue
            params_by_h[h] = _search_hparams(Xtr, ytr, Xval, yval, ELASTICNET_GRID)
            log(f"  [{ref}] h={h}: {params_by_h[h]}")

    # --- Fit each (target, horizon), predict on the NaN-clean test rows ---
    X_test_all = test_df[feat_cols]
    test_valid = ~X_test_all.isna().any(axis=1)
    X_test = X_test_all[test_valid].astype("float32")
    test_meta = test_df.loc[test_valid]
    if len(X_test) == 0:
        raise ValueError("no test rows survive NaN-feature dropping")

    pred_wide = pd.DataFrame(index=X_test.index)
    for col in tcols:
        item, h = parse_target_col(col)
        Xtr, ytr = _xy(train_df, feat_cols, col)
        if val_df is not None:
            Xval, yval = _xy(val_df, feat_cols, col)
            if model_type == "elasticnet":
                params = params_by_h.get(h, {"alpha": 1.0, "l1_ratio": 0.5})
            else:
                params = {}
            X_fit = pd.concat([Xtr, Xval], axis=0)
            y_fit = pd.concat([ytr, yval], axis=0)
        else:
            params = {"alpha": 1.0, "l1_ratio": 0.5} if model_type == "elasticnet" else {}
            X_fit, y_fit = Xtr, ytr
        if len(X_fit) == 0:
            pred_wide[col] = np.nan
            continue
        est = _make_estimator(model_type, **params)
        est.fit(X_fit, y_fit)
        pred_wide[col] = est.predict(X_test)

    return wide_predictions_to_long(test_meta, pred_wide, tcols)
