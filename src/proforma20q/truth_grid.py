"""Canonical truth grid + per-forecast residual arrays.

Ported from the Forma research repo's ``src/eval_cache.py``, minus the on-disk
memmap machinery (the public benchmark works in memory). The math -- key
canonicalization, cell addressing, and ``residual = prediction - actual`` -- is
identical, so metrics match the research pipeline.

The wide truth frame enumerates every scoreable ``(firm, origin)`` row exactly
once. With the discovered target list and the sorted horizon list H, every
scoreable cell has a fixed address::

    row_id = (target_id * len(H) + h_idx) * n_wide + wide_row

so one forecast's prediction errors fit in a single dense float array of length
``n_targets * len(H) * n_wide`` -- NaN wherever the forecast holds no matching
row or the truth actual is NaN. Each (target, horizon) block is a contiguous
slice of length ``n_wide``.

    residual = prediction - actual

equals the change-space residual (baseline cancels), so it depends only on the
forecast and the truth -- never on the comparison pool.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import (
    FIRM_COL,
    HORIZON_COL,
    ORIGIN_COL,
    PREDICTION_COL,
    SIGMA_COL,
    TARGET_COL,
    normalize_columns,
)

LEVEL0_SUFFIX = "_level_0"
_HORIZON_RE = re.compile(r"(.+)_t(\d+)$")


def canon_firm_ids(s: pd.Series) -> pd.Series:
    """Canonicalize firm ids so int and zero-padded-string encodings of the same
    numeric universe match. Cast to nullable Int64 only when lossless for the
    whole series; genuinely non-numeric ids pass through unchanged.
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() == s.notna().sum():
        return num.astype("Int64")
    return s


def canon_origin(s: pd.Series) -> pd.Series:
    """Canonicalize the origin quarter to ``Period[Q]`` -- the calendar quarter
    IS the key, collapsing midnight / EOQ-nanosecond timestamp differences.
    """
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.asfreq("Q") if s.dt.freqstr != "Q-DEC" else s
    return pd.to_datetime(s).dt.to_period("Q")


def discover_targets(truth_df: pd.DataFrame) -> tuple[list[str], dict[str, list[tuple[str, int]]]]:
    """Find ``(target, [(col, h)])`` pairs that also carry a ``{target}_level_0``
    baseline column."""
    per_target: dict[str, list[tuple[str, int]]] = {}
    for col in truth_df.columns:
        m = _HORIZON_RE.match(col)
        if not m:
            continue
        per_target.setdefault(m.group(1), []).append((col, int(m.group(2))))
    targets = sorted(t for t in per_target if f"{t}{LEVEL0_SUFFIX}" in truth_df.columns)
    for t in per_target:
        per_target[t].sort(key=lambda kv: kv[1])
    return targets, per_target


class TruthGrid:
    """Dense cell grid over a wide truth frame (see module docstring)."""

    def __init__(self, truth_df: pd.DataFrame):
        truth_df = normalize_columns(truth_df)
        if FIRM_COL not in truth_df.columns or ORIGIN_COL not in truth_df.columns:
            raise ValueError(
                f"truth frame must have '{FIRM_COL}' and '{ORIGIN_COL}' columns; "
                f"got {list(truth_df.columns)[:8]}...")
        firm_c = canon_firm_ids(truth_df[FIRM_COL])
        origin_c = canon_origin(truth_df[ORIGIN_COL])
        if pd.DataFrame({"f": firm_c, "q": origin_c}).duplicated().any():
            raise ValueError(
                "truth frame has duplicate (firm, origin) rows; the grid requires "
                "unique canonical keys")
        self.truth_df = truth_df

        targets, per_target = discover_targets(truth_df)
        self.targets = targets
        # Unique index for warning-free code lookups (see _canon_codes).
        self._targets_idx = pd.Index(targets)
        self.target_to_id = {t: i for i, t in enumerate(targets)}
        horizons = sorted({h for t in targets for _c, h in per_target[t]})
        self.horizons = horizons
        self.n_h = len(horizons)
        self.n_wide = len(truth_df)
        self.n_cells = len(targets) * self.n_h * self.n_wide

        self._h_lut = np.full((horizons[-1] + 1) if horizons else 1, -1, dtype=np.int64)
        for i, h in enumerate(horizons):
            self._h_lut[h] = i

        self._firm_uniques = pd.Index(firm_c.unique())
        self._q_uniques = pd.Index(origin_c.unique())
        fc = pd.Categorical(firm_c, categories=self._firm_uniques).codes.astype(np.int64)
        qc = pd.Categorical(origin_c, categories=self._q_uniques).codes.astype(np.int64)
        self._wide_lut = np.full((len(self._firm_uniques), len(self._q_uniques)), -1, dtype=np.int64)
        self._wide_lut[fc, qc] = np.arange(self.n_wide, dtype=np.int64)
        self.quarter_codes = qc.astype(np.int32)
        self.n_quarter_codes = len(self._q_uniques)

        self._col_for: dict[tuple[int, int], str] = {}
        for t in targets:
            for col, h in per_target[t]:
                self._col_for[(self.target_to_id[t], int(self._h_lut[h]))] = col
        self._act_cache: dict[int, np.ndarray | None] = {}

    # -- forecast -> cell mapping -------------------------------------------
    @staticmethod
    def _canon_codes(values: pd.Series, canon, categories: pd.Index) -> np.ndarray:
        """Categorical codes of ``canon(values)`` against ``categories``, applying
        the (per-element-expensive) canonicalization only to the unique values."""
        codes, uniq = pd.factorize(values)
        # Index.get_indexer maps each value to its position in ``categories`` (-1
        # when absent), matching Categorical.codes but without the pandas 3.x
        # "values not in categories" deprecation warning (off-grid keys are
        # expected here). Requires ``categories`` to be unique, which it is.
        mapped = categories.get_indexer(canon(pd.Series(uniq))).astype(np.int64)
        out = np.full(len(codes), -1, dtype=np.int64)
        pos = codes >= 0
        out[pos] = mapped[codes[pos]]
        return out

    def map_forecast_rows(self, chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ``(firm, origin, target, horizon) -> row_id`` for one frame.

        Returns ``(row_id int64, valid bool mask)``. Rows whose key is not on the
        grid (unknown firm/origin, unknown target, out-of-range horizon) are
        invalid -- exactly the rows an inner join would drop.
        """
        fcodes = self._canon_codes(chunk[FIRM_COL], canon_firm_ids, self._firm_uniques)
        qcodes = self._canon_codes(chunk[ORIGIN_COL], canon_origin, self._q_uniques)
        tcodes = self._targets_idx.get_indexer(chunk[TARGET_COL]).astype(np.int64)
        h = pd.to_numeric(chunk[HORIZON_COL], errors="coerce").to_numpy(np.float64)
        h_int = np.where(np.isfinite(h), h, -1).astype(np.int64)
        h_in_range = (h_int >= 0) & (h_int < len(self._h_lut))
        h_idx = np.where(h_in_range, self._h_lut[np.clip(h_int, 0, len(self._h_lut) - 1)], -1)

        key_ok = (fcodes >= 0) & (qcodes >= 0)
        wide_row = np.full(len(chunk), -1, dtype=np.int64)
        if key_ok.any():
            wide_row[key_ok] = self._wide_lut[fcodes[key_ok], qcodes[key_ok]]

        valid = (wide_row >= 0) & (tcodes >= 0) & (h_idx >= 0)
        row_id = np.where(valid, (tcodes * self.n_h + h_idx) * self.n_wide + wide_row, -1)
        return row_id, valid

    def actual_for_block(self, block_id: int) -> np.ndarray | None:
        """float64 view of the actual column for ``block_id = target_id*n_h + h_idx``."""
        if block_id in self._act_cache:
            return self._act_cache[block_id]
        col = self._col_for.get((block_id // self.n_h, block_id % self.n_h))
        arr = self.truth_df[col].to_numpy(np.float64, copy=False) if col is not None else None
        self._act_cache[block_id] = arr
        return arr

    def iter_blocks(self):
        """Yield ``(target, horizon, cell_slice, actual_f64, baseline_f64)`` per block."""
        for t_id, target in enumerate(self.targets):
            baseline = self.truth_df[f"{target}{LEVEL0_SUFFIX}"].to_numpy(np.float64, copy=False)
            for h_idx, h in enumerate(self.horizons):
                act = self.actual_for_block(t_id * self.n_h + h_idx)
                if act is None:
                    continue
                start = (t_id * self.n_h + h_idx) * self.n_wide
                yield target, int(h), slice(start, start + self.n_wide), act, baseline


def build_residual(forecast_df: pd.DataFrame, grid: TruthGrid) -> dict:
    """Build the dense residual (and sigma) arrays for one forecast against a grid.

    ``residual = prediction - actual`` (float32; NaN where prediction is NaN,
    actual is NaN, or the key is off-grid). Duplicate keys keep the FIRST
    occurrence in row order. Returns ``{'residual', 'sigma', 'meta'}`` where meta
    records row / dedup / coverage counts.
    """
    forecast_df = normalize_columns(forecast_df)
    residual = np.full(grid.n_cells, np.nan, dtype=np.float32)
    has_sigma = SIGMA_COL in forecast_df.columns
    sigma_arr = np.full(grid.n_cells, np.nan, dtype=np.float32) if has_sigma else None

    row_id, valid = grid.map_forecast_rows(forecast_df)
    n_rows = len(forecast_df)
    n_unmatched = int((~valid).sum())
    n_dup = 0

    if valid.any():
        rid = row_id[valid]
        pred = pd.to_numeric(forecast_df[PREDICTION_COL], errors="coerce").to_numpy(np.float64)[valid]
        sig = (pd.to_numeric(forecast_df[SIGMA_COL], errors="coerce").to_numpy(np.float64)[valid]
               if has_sigma else None)

        # keep='first' across the whole file.
        dup = pd.Series(rid).duplicated().to_numpy()
        if dup.any():
            n_dup = int(dup.sum())
            keep = ~dup
            rid, pred = rid[keep], pred[keep]
            if sig is not None:
                sig = sig[keep]

        blk = rid // grid.n_wide
        wide_row = rid - blk * grid.n_wide
        order = np.argsort(blk, kind="stable")
        rid_s, blk_s, wr_s, pred_s = rid[order], blk[order], wide_row[order], pred[order]
        sig_s = sig[order] if sig is not None else None
        bounds = np.flatnonzero(np.r_[True, blk_s[1:] != blk_s[:-1], True])
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            act_col = grid.actual_for_block(int(blk_s[lo]))
            if act_col is None:
                continue
            residual[rid_s[lo:hi]] = pred_s[lo:hi] - act_col[wr_s[lo:hi]]
            if sig_s is not None:
                sigma_arr[rid_s[lo:hi]] = sig_s[lo:hi]

    meta = {
        "n_file_rows": n_rows,
        "n_unmatched": n_unmatched,
        "n_duplicate": n_dup,
        "n_finite_residual": int(np.isfinite(residual).sum()),
        "has_sigma": has_sigma,
    }
    return {"residual": residual, "sigma": sigma_arr, "meta": meta}
