"""The ProForma-20Q submission schema and its validator.

A submission is a single parquet file, one row per forecasted cell:

    firm      firm identifier (gvkey-derived; matches the truth file's `firm`)
    target    statement item, e.g. "niq"  (one of the pf_full targets)
    origin    the base quarter t (the last quarter observed before forecasting),
              as a quarter-end timestamp or pandas Period[Q]
    horizon   integer h in 1..20 -- the forecast is for quarter t+h
    prediction  point forecast of the item at t+h, IN REGULARIZED SPACE
    sigma     (optional) predictive standard deviation in the SAME regularized
              space -- present only for the probabilistic track (for the
              ``student_t`` family the evaluator reads it as the t scale, not the
              SD; see SUBMISSION.md "Density family")

Everything is in the regularized target space defined by ``transforms.py`` /
``build.py`` (scale -> asinh -> per-item z-score -> clamp). Predictions are
LEVELS at t+h (not changes); the evaluator forms the change internally.

Files must be written with ``engine="fastparquet"`` (never pyarrow) -- see
SUBMISSION.md. For interoperability, the internal Forma column names
(``firm_id`` / ``quarter`` / ``forecast_horizon``) are accepted as aliases and
normalized on read.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import pf_full_targets

PARQUET_ENGINE = "fastparquet"

FIRM_COL = "firm"
TARGET_COL = "target"
ORIGIN_COL = "origin"
HORIZON_COL = "horizon"
PREDICTION_COL = "prediction"
SIGMA_COL = "sigma"

REQUIRED_COLS = [FIRM_COL, TARGET_COL, ORIGIN_COL, HORIZON_COL, PREDICTION_COL]
OPTIONAL_COLS = [SIGMA_COL]
KEY_COLS = [FIRM_COL, TARGET_COL, ORIGIN_COL, HORIZON_COL]

# Internal Forma forecast column names -> public benchmark names.
ALIASES = {
    "firm_id": FIRM_COL,
    "quarter": ORIGIN_COL,
    "forecast_horizon": HORIZON_COL,
}


class SubmissionError(ValueError):
    """Raised when a forecast file does not conform to the submission schema."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any internal-alias columns to the canonical public names.

    Idempotent. Does not copy when there is nothing to rename.
    """
    present = {k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=present) if present else df


def validate_forecast(df: pd.DataFrame, *, strict: bool = True,
                      check_duplicates: bool = True) -> list[str]:
    """Validate a forecast frame against the submission schema.

    Returns a list of human-readable problem strings (empty == valid). With
    ``strict=True`` a non-empty list is raised as :class:`SubmissionError`;
    with ``strict=False`` the caller decides what to do with the warnings.

    Checks: required columns present, every ``target`` is one of the 78 pf_full
    items, horizon is an integer in 1..20, prediction is numeric and not
    entirely null, sigma (if present) is numeric and strictly positive where
    finite, and keys are unique.
    """
    problems: list[str] = []
    df = normalize_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        problems.append(f"missing required column(s): {missing}")
        # Without the required columns the remaining checks are meaningless.
        if strict and problems:
            raise SubmissionError("; ".join(problems))
        return problems

    # target: every value must be one of the 78 pf_full items (a misspelled
    # target would otherwise validate and then silently drop out of the
    # common-sample join in evaluate, scoring on only the rows that matched).
    valid_targets = set(pf_full_targets())
    tgt = df[TARGET_COL].astype(str)
    unknown = sorted(set(tgt[~tgt.isin(valid_targets)]))
    if unknown:
        shown = ", ".join(unknown[:10])
        more = f", ... (+{len(unknown) - 10} more)" if len(unknown) > 10 else ""
        problems.append(
            f"{TARGET_COL} has {len(unknown)} name(s) not among the 78 pf_full "
            f"targets: {shown}{more}")

    # horizon: integer 1..20
    h = pd.to_numeric(df[HORIZON_COL], errors="coerce")
    if h.isna().any():
        problems.append(f"{HORIZON_COL} has {int(h.isna().sum())} non-integer / null value(s)")
    else:
        if (h != h.round()).any():
            problems.append(f"{HORIZON_COL} must be whole numbers")
        lo, hi = h.min(), h.max()
        if lo < 1 or hi > 20:
            problems.append(f"{HORIZON_COL} out of range [1, 20]: observed [{lo}, {hi}]")

    # prediction numeric and not all null
    pred = pd.to_numeric(df[PREDICTION_COL], errors="coerce")
    if pred.notna().sum() == 0:
        problems.append(f"{PREDICTION_COL} column is entirely null / non-numeric")

    # sigma (optional): numeric, > 0 where finite
    if SIGMA_COL in df.columns:
        sig = pd.to_numeric(df[SIGMA_COL], errors="coerce")
        finite = sig.notna()
        if finite.any() and (sig[finite] <= 0).any():
            n_bad = int((sig[finite] <= 0).sum())
            problems.append(f"{SIGMA_COL} must be > 0 where present ({n_bad} non-positive value(s))")

    # duplicate keys (evaluator keeps first; warn so it isn't silent). Skippable
    # because a streamed write validates each block separately, and blocks are
    # disjoint by construction -- a full-coverage submission is ~550M rows and
    # the duplicate scan over object keys is the expensive check.
    if check_duplicates:
        dup = df.duplicated(subset=KEY_COLS)
        if dup.any():
            problems.append(
                f"{int(dup.sum())} duplicate {KEY_COLS} row(s) (evaluator keeps the first of each)")

    if strict and problems:
        raise SubmissionError("; ".join(problems))
    return problems


def read_forecast(path, *, validate: bool = True, strict: bool = False) -> pd.DataFrame:
    """Read a forecast parquet with the canonical engine and normalize columns.

    With ``validate=True`` the schema check runs; ``strict`` controls whether
    problems raise or are attached (printed) as a warning.
    """
    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    df = normalize_columns(df)
    if validate:
        problems = validate_forecast(df, strict=strict)
        if problems and not strict:
            print(f"  Warning: {path}: {'; '.join(problems)}")
    return df


def _forecast_payload(df: pd.DataFrame) -> pd.DataFrame:
    """Submission columns only, with float32 predictions and string keys."""
    out = df[[c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]].copy()
    for col in (PREDICTION_COL, SIGMA_COL):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    # ids may be carried internally as categorical / integer codes to keep the
    # in-memory footprint down; the on-disk schema is plain strings.
    for col in (FIRM_COL, TARGET_COL):
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out


def write_forecast(df: pd.DataFrame, path, *, validate: bool = True) -> None:
    """Write a forecast frame to parquet with the canonical fastparquet engine.

    Downcasts ``prediction``/``sigma`` to float32 (lossless for the ~3-4 sig-fig
    metrics in regularized space) to match the Forma forecast-write contract.

    Needs the whole forecast in memory. A **full-coverage** submission on the
    canonical test split is 352,106 origins x 78 targets x 20 horizons =
    **549,285,360 rows (~4 GB on disk, >20 GB in memory)** -- use
    :func:`write_forecast_blocks` for anything near that size.
    """
    df = normalize_columns(df)
    if validate:
        validate_forecast(df, strict=True)
    _forecast_payload(df).to_parquet(path, engine=PARQUET_ENGINE, index=False)


def write_forecast_blocks(blocks: Iterable[pd.DataFrame], path, *,
                          validate: bool = True,
                          rows_per_group: int = 4_000_000) -> int:
    """Stream a forecast to parquet, appending row-groups instead of one write.

    ``blocks`` is any iterable of submission-schema frames -- typically one per
    ``(target, horizon)``. Blocks are buffered to ~``rows_per_group`` rows and
    written as successive parquet row-groups, so peak memory is set by the
    buffer (a few hundred MB) rather than by the size of the submission.

    This is the writer a full-coverage entry needs: ~550M rows will not fit in a
    single frame on an ordinary machine.

    Validation runs per block; the cross-block duplicate-key scan is skipped
    (blocks are disjoint by construction) -- run ``proforma20q validate`` on the
    finished file for the complete check.

    Returns the number of rows written.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    buf: list[pd.DataFrame] = []
    buffered = n_rows = 0
    first = True

    def flush():
        nonlocal buf, buffered, first
        if not buf:
            return
        part = buf[0] if len(buf) == 1 else pd.concat(buf, ignore_index=True)
        part.to_parquet(path, engine=PARQUET_ENGINE, index=False, append=not first)
        first = False
        buf, buffered = [], 0

    for block in blocks:
        block = normalize_columns(block)
        if validate:
            validate_forecast(block, strict=True, check_duplicates=False)
        payload = _forecast_payload(block)
        if payload.empty:
            continue
        buf.append(payload)
        buffered += len(payload)
        n_rows += len(payload)
        if buffered >= rows_per_group:
            flush()
    flush()
    if first:
        raise SubmissionError(f"no forecast blocks to write to {path}")
    return n_rows
