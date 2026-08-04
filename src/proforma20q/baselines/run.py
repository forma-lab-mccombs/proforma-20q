"""Driver: build the reference-baseline forecast files from the tabular splits."""
from __future__ import annotations

from pathlib import Path

from ..schema import FIRM_COL, ORIGIN_COL, write_forecast_blocks
from .common import (discover_targets, feature_columns, load_tabular, tabular_columns,
                     target_columns)

# What each baseline actually reads. `naive` needs the four seasonal-alignment
# levels (`level_0..3`); `fade` needs `level_0` plus the future-target columns;
# the linear family needs the whole feature matrix. Reading a split in full
# costs ~6 GB, so the difference decides whether `baselines --which naive,fade`
# runs on an ordinary machine.
_NEEDS_FIT_SPLITS = {"fade", "elasticnet", "linear"}


def _resolve_split_paths(processed_dir, suffix) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    return {
        split: processed_dir / f"tabular_{split}__{suffix}.parquet"
        for split in ("train", "val", "test")
    }


def _columns_for(which, cols, targets) -> list[str]:
    """Union of the columns the requested baselines read."""
    from .naive import N_LEVEL_LAGS

    cols_set = set(cols)
    need = {FIRM_COL, ORIGIN_COL}
    need |= {f"{t}_level_0" for t in targets if f"{t}_level_0" in cols_set}
    if "naive" in which:
        # seasonal random walk: h=1 <- level_3, h=2 <- level_2, ...
        need |= {f"{t}_level_{lag}" for t in targets for lag in range(N_LEVEL_LAGS)
                 if f"{t}_level_{lag}" in cols_set}
    if any(n != "naive" for n in which):
        need |= set(target_columns(cols, targets))
    if any(n in ("elasticnet", "linear") for n in which):
        need |= set(feature_columns(cols))
    return [c for c in cols if c in need]


def run_baselines(
    processed_dir,
    suffix: str,
    output_dir,
    *,
    which=None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Fit each requested baseline and write ``{model}__predictions.parquet``.

    Args:
        processed_dir: directory holding ``tabular_{split}__{suffix}.parquet``.
        suffix: dataset suffix, e.g. ``pf_full__r13_node_optionD_indfe_val8``.
        output_dir: where forecast parquets are written.
        which: iterable of baseline names (default: all of ``BASELINES``).
        verbose: progress printing.

    Returns:
        ``{model_name: forecast_path}``.
    """
    from . import BASELINES, iter_baseline_blocks

    which = list(which) if which is not None else list(BASELINES)
    paths = _resolve_split_paths(processed_dir, suffix)
    for split, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing tabular split {p}. Run `proforma20q build` first.")

    log = print if verbose else (lambda *a, **k: None)
    # Decide what to read from the parquet schema, before reading any data.
    all_cols = tabular_columns(paths["test"])
    targets = discover_targets(all_cols)
    cols = _columns_for(which, all_cols, targets)
    log(f"Loading tabular splits for suffix '{suffix}' "
        f"({len(cols)} of {len(all_cols)} columns for {', '.join(which)})...")
    test_df = load_tabular(paths["test"], columns=cols)
    if any(n in _NEEDS_FIT_SPLITS for n in which):
        train_df = load_tabular(paths["train"], columns=cols)
        val_df = load_tabular(paths["val"], columns=cols)
    else:
        train_df = val_df = None      # `naive` never looks at the fit splits
    fit_rows = "" if train_df is None else \
        f"train={len(train_df):,} val={len(val_df):,} "
    log(f"  {len(targets)} targets; {fit_rows}test={len(test_df):,} rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    # What to forecast comes from the file's schema, not from the projected read:
    # `naive` never loads a truth column but still predicts every one of them.
    tcols = target_columns(all_cols, targets)
    n_h = len({c.rsplit("_t", 1)[-1] for c in tcols})
    log(f"  each baseline forecasts up to {len(test_df) * len(tcols):,} cells "
        f"({len(test_df):,} origins x {len(targets)} targets x {n_h} horizons)")
    for name in which:
        log(f"\n=== baseline: {name} ===")
        # Streamed: one (target, horizon) block is assembled, validated and
        # appended as a parquet row-group at a time. Building the whole forecast
        # first is what OOMs at benchmark scale.
        blocks = iter_baseline_blocks(name, train_df, test_df, val_df=val_df,
                                      targets=targets, target_cols=tcols,
                                      verbose=verbose)
        path = output_dir / f"{name}__predictions.parquet"
        n = write_forecast_blocks(blocks, path)
        out[name] = path
        log(f"  wrote {n:,} rows -> {path}")
    return out
