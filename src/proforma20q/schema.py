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

import numpy as np
import pandas as pd

from .config import pf_full_targets

PARQUET_ENGINE = "fastparquet"

FIRM_COL = "firm"
TARGET_COL = "target"
ORIGIN_COL = "origin"
HORIZON_COL = "horizon"
PREDICTION_COL = "prediction"
SIGMA_COL = "sigma"

# The build clamps every regularized value to |z| <= max_abs_zscore. A forecast
# is allowed to overshoot a little (a model may extrapolate past the clamp), but
# an order of magnitude past it means the submission is not in this space at all.
_MAX_ABS_Z = 6.0
_OUT_OF_RANGE_FACTOR = 10.0

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
                      check_duplicates: bool = True,
                      check_all_null: bool = True) -> list[str]:
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
    #
    # Checked over the DISTINCT names, never the column. `.astype(str)` builds a
    # fixed-width unicode array -- 13 chars x 4 bytes x 472.7M rows = 22.9 GiB on
    # a full-coverage submission, however compactly the column is stored. The
    # check is over a set of at most a few dozen names; it costs O(distinct).
    valid_targets = set(pf_full_targets())
    tgt = df[TARGET_COL]
    if isinstance(tgt.dtype, pd.CategoricalDtype):
        names = tgt.cat.categories          # free: already the distinct values
    else:
        names = pd.unique(tgt)
    unknown = sorted({str(x) for x in names} - valid_targets)
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

    # null keys: a missing firm / target joins to nothing and is silently dropped
    # by the evaluator, so it must not pass as a valid row.
    for col in (FIRM_COL, TARGET_COL, ORIGIN_COL, HORIZON_COL):
        n_null = int(df[col].isna().sum())
        if n_null:
            problems.append(f"{col} has {n_null} null value(s); key columns must be complete")

    # firm must be the truth file's gvkey STRING ("001045"). An integer 1045
    # validates as a plausible id and then matches nothing at all. Checked on the
    # inferred CONTENT, not just the dtype: an object column can hold Python
    # ints, which is exactly what a frame built in memory (rather than read from
    # parquet) tends to produce.
    firm = df[FIRM_COL]
    if isinstance(firm.dtype, pd.CategoricalDtype):
        firm = pd.Series(firm.cat.categories)
    inferred = pd.api.types.infer_dtype(firm, skipna=True)
    if inferred not in ("string", "unicode", "empty"):
        problems.append(
            f"{FIRM_COL} must be the gvkey string (e.g. '001045'); this column "
            f"holds {inferred} values, which will not join to the truth file")

    # prediction numeric and not all null. ``check_all_null`` is off when
    # validating one block of a streamed write: a single (target, horizon) with
    # no fittable training rows is legitimately all-NaN, and only the whole
    # submission being empty is an error.
    pred = pd.to_numeric(df[PREDICTION_COL], errors="coerce")
    if check_all_null and pred.notna().sum() == 0:
        problems.append(f"{PREDICTION_COL} column is entirely null / non-numeric")

    # sigma (optional): numeric, > 0 where finite
    if SIGMA_COL in df.columns:
        sig = pd.to_numeric(df[SIGMA_COL], errors="coerce")
        finite = sig.notna()
        if finite.any() and (sig[finite] <= 0).any():
            n_bad = int((sig[finite] <= 0).sum())
            problems.append(f"{SIGMA_COL} must be > 0 where present ({n_bad} non-positive value(s))")

    # duplicate keys (evaluator keeps first; warn so it isn't silent).
    # ``check_duplicates`` is off only for a whole-file scan that has already
    # checked each row-group; a streamed write still checks every block, because
    # duplicates come from a repeated key WITHIN a block (the origin frame having
    # a duplicated firm-quarter), not from across blocks.
    if check_duplicates:
        dup = df.duplicated(subset=KEY_COLS)
        if dup.any():
            problems.append(
                f"{int(dup.sum())} duplicate {KEY_COLS} row(s) (evaluator keeps the first of each)")

    if strict and problems:
        raise SubmissionError("; ".join(problems))
    return problems


def forecast_warnings(df: pd.DataFrame) -> list[str]:
    """Advisory checks: schema-valid files that will still score badly or hurt
    others. Separate from :func:`validate_forecast` because none of these make a
    file invalid -- partial coverage is explicitly allowed.
    """
    warnings: list[str] = []
    if PREDICTION_COL not in df.columns:
        return warnings
    arr = pd.to_numeric(df[PREDICTION_COL], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan)
    if not arr.size:
        return warnings

    # Non-finite predictions do not just cost YOU coverage: `evaluate` scores on
    # the strict all-models common sample, so a submission that is 40% inf
    # silently shrinks the scored sample for EVERY model it is compared against
    # (demonstrated in the audit: 16,000 -> 9,600 cells).
    n_bad = int((~np.isfinite(arr)).sum())
    if n_bad:
        warnings.append(
            f"{n_bad:,} of {arr.size:,} predictions ({n_bad / arr.size:.1%}) are "
            "non-finite; they drop out of the common sample, shrinking it for "
            "every model scored alongside yours")

    # Predictions live in the regularized space (|z| <= 6). A submission in raw
    # currency units validates as numeric and then scores as noise -- the most
    # likely newcomer error, so name it instead of letting it through.
    finite = arr[np.isfinite(arr)]
    if finite.size:
        worst = float(np.abs(finite).max())
        if worst > _MAX_ABS_Z * _OUT_OF_RANGE_FACTOR:
            n_out = int((np.abs(finite) > _MAX_ABS_Z).sum())
            warnings.append(
                f"predictions reach |{worst:.4g}|, far outside the regularized range "
                f"|z| <= {_MAX_ABS_Z:g} ({n_out:,} value(s) beyond it). Forecasts must be "
                "in the build's regularized space, not raw currency units -- see "
                "SUBMISSION.md")
    return warnings


def validate_forecast_file(path, *, strict: bool = False) -> tuple[list[str], int]:
    """Validate a forecast parquet **without loading it**, row-group by row-group.

    A full-coverage submission is ~550M rows; materialized as a frame that is
    ~73 GB (the ``firm`` / ``target`` object columns dominate), so the check has
    to stream. Each row-group is validated in full, including its own
    duplicate-key scan.

    **Limitation, stated rather than hidden:** duplicate keys spanning two
    row-groups are not detected -- that would need a 550M-key index. Writers that
    emit one row-group per ``(target, horizon)`` block cannot produce them.

    Returns ``(problems, n_rows)``. Advisory warnings (coverage, out-of-range
    magnitudes) are returned by :func:`forecast_file_warnings`.
    """
    problems, n_rows, _warn = scan_forecast_file(path)
    if strict and problems:
        raise SubmissionError("; ".join(problems))
    return problems, n_rows


def forecast_file_warnings(path) -> list[str]:
    """Advisory warnings for a forecast file, accumulated over its row-groups."""
    return scan_forecast_file(path)[2]


def scan_forecast_file(path) -> tuple[list[str], int, list[str]]:
    """One streaming pass over a forecast file: ``(problems, n_rows, warnings)``.

    The public entry point for checking a whole file. :func:`validate_forecast_file`
    and :func:`forecast_file_warnings` are conveniences over it, but calling both
    scans a ~550M-row submission twice -- which is exactly what streaming exists
    to avoid. Prefer this.
    """
    import fastparquet  # noqa: PLC0415

    pf = fastparquet.ParquetFile(str(path))
    problems: list[str] = []
    n_rows = 0
    n_nonfinite = 0
    worst = 0.0
    n_out_of_range = 0
    for i, group in enumerate(pf.iter_row_groups()):
        group = normalize_columns(group)
        n_rows += len(group)
        found = validate_forecast(group, strict=False, check_all_null=False)
        for p in found:
            msg = f"row-group {i}: {p}"
            if msg not in problems:
                problems.append(msg)
        if PREDICTION_COL in group.columns:
            arr = pd.to_numeric(group[PREDICTION_COL], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan)
            ok = np.isfinite(arr)
            n_nonfinite += int((~ok).sum())
            if ok.any():
                worst = max(worst, float(np.abs(arr[ok]).max()))
                n_out_of_range += int((np.abs(arr[ok]) > _MAX_ABS_Z).sum())

    warnings: list[str] = []
    if n_rows == 0:
        problems.append("file contains no rows")
    elif n_nonfinite == n_rows:
        problems.append(f"{PREDICTION_COL} is null / non-finite in every row")
    elif n_nonfinite:
        warnings.append(
            f"{n_nonfinite:,} of {n_rows:,} predictions ({n_nonfinite / n_rows:.1%}) are "
            "non-finite; they drop out of the common sample, shrinking it for every "
            "model scored alongside yours")
    if worst > _MAX_ABS_Z * _OUT_OF_RANGE_FACTOR:
        warnings.append(
            f"predictions reach |{worst:.4g}|, far outside the regularized range "
            f"|z| <= {_MAX_ABS_Z:g} ({n_out_of_range:,} value(s) beyond it). Forecasts "
            "must be in the build's regularized space, not raw currency units -- see "
            "SUBMISSION.md")
    return problems, n_rows, warnings


def read_forecast(path, *, validate: bool = True, strict: bool = False) -> pd.DataFrame:
    """Read a forecast parquet with the canonical engine and normalize columns.

    With ``validate=True`` the schema check runs; ``strict`` controls whether
    problems raise or are attached (printed) as a warning.
    """
    df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    df = normalize_columns(df)
    # Read side of the same normalization: a file written by another tool may
    # still carry a second- or millisecond-resolution origin.
    if ORIGIN_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[ORIGIN_COL]):
        df[ORIGIN_COL] = df[ORIGIN_COL].astype("datetime64[ns]")
    if validate:
        problems = validate_forecast(df, strict=strict)
        if problems and not strict:
            print(f"  Warning: {path}: {'; '.join(problems)}")
        for w in forecast_warnings(df):
            print(f"  Warning: {path}: {w}")
    return df


def _forecast_payload(df: pd.DataFrame) -> pd.DataFrame:
    """Submission columns only, with a canonical dtype for every column.

    Every write goes through here, so a streamed write cannot drift between
    blocks: the parquet schema is fixed by the first row-group, and a later
    block with a different ``origin`` representation would otherwise be coerced
    against it silently (Period ordinals read back as 1970-era nanoseconds).
    """
    out = df[[c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]].copy()

    # origin: Period[Q] is an accepted input form; the on-disk form is the
    # quarter-end timestamp, matching the truth file's `quarter`.
    #
    # The `[ns]` cast is load-bearing, not tidiness. `pd.Timestamp("2011-12-31")`
    # is `datetime64[s]` under pandas >= 2.2; fastparquet stores that as
    # TIMESTAMP[MILLIS] while recording `numpy_type: datetime64[s]`, and pandas
    # then refuses the lossy read with
    #     Cannot losslessly cast '1325289 ms' to s
    # -- so the SUBMISSION.md example wrote a file this package could not read
    # back. Every fixture in the suite happened to build origins via
    # `period_range(...).to_timestamp(how="end")`, which is nanosecond EOQ and
    # dodges it, which is why CI stayed green.
    origin = out[ORIGIN_COL]
    if isinstance(origin.dtype, pd.PeriodDtype):
        origin = origin.dt.to_timestamp(how="end")
    else:
        origin = pd.to_datetime(origin)
    out[ORIGIN_COL] = origin.astype("datetime64[ns]")

    if HORIZON_COL in out.columns:
        # Unconditional, so the "no drift between blocks" guarantee actually
        # holds: a conditional cast would let one block with a null horizon write
        # float64 and silently fix the file's schema as float.
        if not out[HORIZON_COL].notna().all():
            raise SubmissionError(
                f"{HORIZON_COL} has {int(out[HORIZON_COL].isna().sum())} null "
                "value(s); it is a key column and must be a whole number 1..20")
        out[HORIZON_COL] = out[HORIZON_COL].astype("int64")

    for col in (PREDICTION_COL, SIGMA_COL):
        if col in out.columns:
            before = pd.to_numeric(out[col], errors="coerce")
            out[col] = before.astype("float32")
            if col == SIGMA_COL:
                # float32 underflow would turn a valid tiny sigma into 0 and the
                # file would fail the check it just passed on the way in.
                lost = int(((before > 0) & (out[col] == 0)).sum())
                if lost:
                    raise SubmissionError(
                        f"{lost} {SIGMA_COL} value(s) underflow to 0 in float32; "
                        "rescale them before writing")

    # ids may be carried internally as categorical codes to keep the in-memory
    # footprint down; the on-disk schema is plain strings. `astype(object)`, not
    # `astype(str)` -- the latter renders a missing category as the literal
    # "nan", which then validates as a real firm id that joins to nothing.
    for col in (FIRM_COL, TARGET_COL):
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)
    return out


def write_forecast(df: pd.DataFrame, path, *, validate: bool = True) -> None:
    """Write a forecast frame to parquet with the canonical fastparquet engine.

    Downcasts ``prediction``/``sigma`` to float32 (lossless for the ~3-4 sig-fig
    metrics in regularized space) to match the Forma forecast-write contract.

    Needs the whole forecast in memory. A **full-coverage** submission on the
    canonical test split is 352,962 origins x 78 targets x 20 horizons =
    **550,620,720 rows (~4 GB on disk, >20 GB in memory)** -- use
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

    Every block is validated, including its own duplicate-key scan; only the
    *cross-block* comparison is out of reach, and blocks are disjoint by
    construction. A block whose predictions are entirely null is allowed (one
    ``(target, horizon)`` can legitimately have no fittable training rows) --
    the all-null check is applied to the submission as a whole instead.

    Rows accumulate in ``<path>.partial`` and are renamed into place only on
    success, so a run that dies at block 1,400 of 1,560 cannot leave behind a
    well-formed parquet that scores as an intentional partial-coverage entry --
    and cannot destroy a good submission from a previous run.

    Returns the number of rows written.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    buf: list[pd.DataFrame] = []
    buffered = n_rows = 0
    first = True
    any_finite = False

    def flush():
        nonlocal buf, buffered, first
        if not buf:
            return
        part = buf[0] if len(buf) == 1 else pd.concat(buf, ignore_index=True)
        part.to_parquet(tmp, engine=PARQUET_ENGINE, index=False, append=not first)
        first = False
        buf, buffered = [], 0

    try:
        for block in blocks:
            block = normalize_columns(block)
            missing = [c for c in REQUIRED_COLS if c not in block.columns]
            if missing:
                raise SubmissionError(f"block is missing required column(s): {missing}")
            payload = _forecast_payload(block)
            if payload.empty:
                continue
            if validate:
                validate_forecast(payload, strict=True, check_all_null=False)
            any_finite = any_finite or bool(
                np.isfinite(payload[PREDICTION_COL].to_numpy()).any())
            buf.append(payload)
            buffered += len(payload)
            n_rows += len(payload)
            if buffered >= rows_per_group:
                flush()
        flush()
        if first:
            raise SubmissionError(f"no forecast blocks to write to {path}")
        if validate and not any_finite:
            raise SubmissionError(
                f"every {PREDICTION_COL} across all {n_rows:,} rows is null / non-finite")
    except BaseException:
        if tmp.exists():
            print(f"  incomplete forecast left at {tmp} ({n_rows:,} rows written); "
                  f"{path} was not modified")
        raise
    tmp.replace(path)
    return n_rows
