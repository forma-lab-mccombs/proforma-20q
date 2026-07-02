"""Driver: build the reference-baseline forecast files from the tabular splits."""
from __future__ import annotations

from pathlib import Path

from ..schema import write_forecast
from .common import discover_targets, load_tabular


def _resolve_split_paths(processed_dir, suffix) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    return {
        split: processed_dir / f"tabular_{split}__{suffix}.parquet"
        for split in ("train", "val", "test")
    }


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
    from . import BASELINES, run_baseline

    which = list(which) if which is not None else list(BASELINES)
    paths = _resolve_split_paths(processed_dir, suffix)
    for split, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing tabular split {p}. Run `proforma20q build` first.")

    log = print if verbose else (lambda *a, **k: None)
    log(f"Loading tabular splits for suffix '{suffix}'...")
    train_df = load_tabular(paths["train"])
    val_df = load_tabular(paths["val"])
    test_df = load_tabular(paths["test"])
    targets = discover_targets(test_df)
    log(f"  {len(targets)} targets; train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in which:
        log(f"\n=== baseline: {name} ===")
        fc = run_baseline(name, train_df, test_df, val_df=val_df, targets=targets, verbose=verbose)
        path = output_dir / f"{name}__predictions.parquet"
        write_forecast(fc, path)
        out[name] = path
        log(f"  wrote {len(fc):,} rows -> {path}")
    return out
