"""The ProForma-20Q evaluation engine.

Faithful standalone port of the Forma research repo's ``src/evaluate.py``
scoring semantics. The numbers match the research pipeline to float precision;
this is the metric the paper and the leaderboard report.

Protocol (all in the regularized target space of ``transforms.py``):

* **Change space.** For a cell ``(firm, origin=t, target, horizon=h)``:
  ``truth_change = a_{t+h} - a_t`` and ``pred_change = pred - a_t``. The
  residual ``pred_change - truth_change == pred - a_{t+h}`` -- the baseline
  ``a_t`` cancels, so the residual depends only on the forecast and the truth.
* **R^2** is against the variance of *realized changes* per (target, horizon)
  block, with the denominator SHARED across all compared models (model-
  independent). MAE / MSE are identical in levels and changes.
* **Common-sample inner join.** A cell scores for everyone only if the truth,
  the baseline, AND every compared model's prediction are all finite -- the
  strict all-models intersection. A post-loop self-check asserts each model's
  per-block ``n_metric`` equals the block's ``n_complete``.
* **Aggregation levels:** ``global`` / ``by_target`` / ``by_horizon`` /
  ``by_target_horizon``.
* **Probabilistic track** (when a forecast carries ``sigma``): mean NLL and mean
  CRPS under the forecast's own density family (gaussian / laplace / student_t),
  plus the Gaussian calibration diagnostics ``z2`` (mean standardized squared
  residual, -> 1 when calibrated) and ``cover95`` (central-95% coverage).

Mixture NLL, PIT histograms, and Diebold-Mariano tests are extended in the main
research repo (publication plan Sec 4.4-4.5); port them here once they land,
tracking the main repo rather than forking -- see SUBMISSION.md / README.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr  # standard-normal CDF (vectorized)

from .schema import SIGMA_COL, normalize_columns, read_forecast
from .truth_grid import TruthGrid, build_residual

# Aggregation levels (group keys); empty list == the global pool.
AGG_LEVELS: dict[str, list[str]] = {
    "by_target_horizon": ["target", "horizon"],
    "by_target": ["target"],
    "by_horizon": ["horizon"],
    "global": [],
}

# central-95% z multiplier (two-sided Gaussian)
_Z95 = 1.959963984540054


# --------------------------- density families --------------------------- #

def nll_per_entry(res: np.ndarray, sigma: np.ndarray, family: str, df: float | None) -> np.ndarray:
    """Per-observation NLL of the actual under the predictive density centered at
    the forecast. ``res = pred - actual``; ``sigma`` is the predictive std in the
    same regularized space. Full normalizing constant included (a proper mean NLL).
    """
    sigma_sq = sigma * sigma
    log_sigma_sq = np.log(sigma_sq)
    if family == "gaussian":
        return 0.5 * math.log(2.0 * math.pi) + 0.5 * log_sigma_sq + 0.5 * res * res / sigma_sq
    if family == "laplace":
        # sigma is the predictive STD; Laplace scale b = sigma / sqrt(2).
        b = sigma / math.sqrt(2.0)
        return np.log(2.0 * b) + np.abs(res) / b
    if family == "student_t":
        if df is None:
            raise ValueError("student_t NLL requires 'df' (degrees of freedom)")
        nu = float(df)
        c = math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
        return -c + 0.5 * log_sigma_sq + 0.5 * (nu + 1.0) * np.log1p(res * res / (nu * sigma_sq))
    raise ValueError(f"Unknown NLL family {family!r} (gaussian|laplace|student_t)")


def crps_per_entry(res: np.ndarray, sigma: np.ndarray, family: str, df: float | None) -> np.ndarray:
    """Per-observation CRPS under the predictive density (closed forms). A proper
    score in target units, far less tail-sensitive than NLL. Unwired families
    (student_t) return NaN so they drop out of the accumulator cleanly.
    """
    if family == "gaussian":
        w = res / sigma
        phi = np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        return sigma * (w * (2.0 * ndtr(w) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    if family == "laplace":
        b = sigma / math.sqrt(2.0)
        z = np.abs(res) / b
        return b * (z + np.exp(-z) - 0.75)
    if family == "student_t":
        return np.full(np.shape(res), np.nan, dtype=np.float64)
    raise ValueError(f"Unknown CRPS family {family!r} (gaussian|laplace|student_t)")


def _read_family_sidecar(path) -> tuple[str, float | None]:
    """Resolve (family, df) from an optional ``{stem}.nll.json`` sidecar next to a
    forecast file. Absent/unparseable -> ('gaussian', None), the default track.
    """
    side = Path(path).parent / f"{Path(path).stem}.nll.json"
    if side.exists():
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
            fam = str(meta.get("family", "gaussian")).lower()
            dfv = meta.get("df", None)
            return fam, (float(dfv) if dfv is not None else None)
        except Exception as e:  # noqa: BLE001
            print(f"  Warning: could not parse {side.name}: {e}; defaulting to gaussian")
    return "gaussian", None


# --------------------------- accumulators --------------------------- #

def _new_truth_entry() -> dict:
    return {"quarters": set(), "n_complete": 0, "sum_y": 0.0, "sum_y2": 0.0}


def _new_model_entry() -> dict:
    return {
        "n_obs": 0, "n_metric": 0, "ss_res": 0.0, "sae": 0.0,
        "sum_nll": 0.0, "n_nll": 0, "sum_crps": 0.0, "n_crps": 0,
        "sum_z2": 0.0, "n_z2": 0, "sum_cover": 0.0, "n_cover": 0,
    }


def _group_vals(level: str, target: str, horizon: int) -> tuple:
    if level == "global":
        return ()
    if level == "by_target":
        return (target,)
    if level == "by_horizon":
        return (horizon,)
    if level == "by_target_horizon":
        return (target, horizon)
    raise ValueError(f"unknown level {level!r}")


@dataclass
class EvalResult:
    """Structured evaluation output.

    ``metrics`` maps each aggregation level to a tidy DataFrame with columns
    ``model``, the group keys for that level (``target`` / ``horizon``), and the
    metric columns ``r2, mae, mse, nll, crps, z2, cover95, avg_obs,
    n_complete``. ``n_common`` is the total common-sample size; ``invariant_ok``
    is the coverage-identity self-check.
    """
    metrics: dict[str, pd.DataFrame]
    n_common: int
    invariant_ok: bool
    dropped_targets: list[str] = field(default_factory=list)

    def leaderboard(self, metric: str = "r2") -> pd.DataFrame:
        """One row per model of the global-pool metrics, sorted best-first for
        ``metric`` (R^2/coverage: higher better; error/score: lower better)."""
        g = self.metrics["global"]
        if g.empty:
            return g
        ascending = metric in {"mae", "mse", "nll", "crps"}
        cols = ["model", "r2", "mae", "mse", "nll", "crps", "z2", "cover95", "avg_obs", "n_complete"]
        cols = [c for c in cols if c in g.columns]
        out = g[cols].copy()
        if metric in out.columns:
            out = out.sort_values(metric, ascending=ascending, na_position="last")
        return out.reset_index(drop=True)

    def write_csvs(self, output_dir) -> list[Path]:
        """Write one CSV per (level, metric) plus the global leaderboard."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        metric_cols = ["r2", "mae", "mse", "nll", "crps", "z2", "cover95", "avg_obs"]
        for level, keys in AGG_LEVELS.items():
            df = self.metrics.get(level)
            if df is None or df.empty:
                continue
            for metric in metric_cols:
                if metric not in df.columns or df[metric].notna().sum() == 0:
                    continue
                if keys:
                    pivot = df.pivot_table(index=keys, columns="model", values=metric)
                else:
                    pivot = df.pivot_table(columns="model", values=metric, aggfunc="first")
                if pivot.empty:
                    continue
                p = output_dir / f"{metric}__{level}.csv"
                pivot.to_csv(p)
                written.append(p)
        lb = self.leaderboard()
        if not lb.empty:
            p = output_dir / "leaderboard.csv"
            lb.to_csv(p, index=False)
            written.append(p)
        return written


def evaluate_forecasts(
    forecasts: dict[str, pd.DataFrame] | list,
    truth: pd.DataFrame,
    *,
    families: dict[str, tuple[str, float | None]] | None = None,
    allow_missing: bool = False,
    sample_mask: pd.DataFrame | str | None = None,
    verbose: bool = True,
) -> EvalResult:
    """Score one or more forecasts against a wide truth frame on the all-models
    common sample.

    Args:
        forecasts: either a mapping ``{model_name: forecast_df}`` or a list whose
            items are ``model_name`` (path resolved elsewhere), ``(model_name,
            df)`` tuples, or ``(model_name, path)`` tuples. Forecast frames use
            the submission schema (``firm, target, origin, horizon, prediction,
            [sigma]``; internal Forma aliases accepted).
        truth: wide truth frame -- the ``tabular_test`` artifact: ``firm``,
            ``origin`` (aka ``firm_id`` / ``quarter``), ``{target}_level_0``, and
            ``{target}_t{h}`` columns, all in regularized space.
        families: optional ``{model_name: (family, df)}`` overriding the density
            family per model (default gaussian). Path inputs also read a
            ``{stem}.nll.json`` sidecar.
        allow_missing: if a forecast fails to read/parse, score only the readable
            ones instead of raising (this re-bases the common sample!).
        sample_mask: optional restriction of the scored cells to a fixed pooled
            sample -- a frame (or parquet path) of ``firm, target, origin,
            horizon`` keys (aliases accepted). Only cells whose key is in the mask
            are scored; every metric is then computed on ``mask INTERSECT the
            all-models common sample``. Reproduces a paper's pooled column (e.g.
            the 327.2M-cell Full sample = the binding model's coverage) without
            needing that model's forecast. Keys off-grid are ignored.
        verbose: print progress + coverage diagnostics.

    Returns:
        :class:`EvalResult`.
    """
    families = dict(families or {})
    log = print if verbose else (lambda *a, **k: None)

    grid = TruthGrid(normalize_columns(truth))
    if not grid.targets:
        raise ValueError("truth frame has no scoreable {target}_t{h} / {target}_level_0 columns")

    # Optional pooled-sample restriction: a dense cell bitmask over the grid.
    # Two accepted forms: a keys frame/parquet (firm/target/origin/horizon --
    # portable, matches by value) or a grid-aligned bit array / ``.npy``
    # (``np.packbits`` of the canonical cell order -- compact, no firm ids, but
    # requires the truth to be the canonical ``tabular_test``).
    mask_bits = None
    if sample_mask is not None:
        mask_bits = _resolve_sample_mask(sample_mask, grid, log)

    # ---- Phase 1: residual arrays per model ----
    residuals: dict[str, dict] = {}
    sigma_family: dict[str, tuple[str, float | None]] = {}
    failed: list[str] = []
    items = forecasts.items() if isinstance(forecasts, dict) else _iter_forecast_list(forecasts)
    for name, obj in items:
        try:
            if isinstance(obj, (str, Path)):
                fam = families.get(name) or _read_family_sidecar(obj)
                fdf = read_forecast(obj, validate=True, strict=False)
            else:
                fam = families.get(name, ("gaussian", None))
                fdf = normalize_columns(obj)
            cache = build_residual(fdf, grid)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{name} ({e})")
            continue
        residuals[name] = cache
        if cache["sigma"] is not None:
            sigma_family[name] = fam

    if failed:
        msg = (f"{len(failed)} forecast(s) failed to read (dropping them re-bases the "
               f"all-models common sample): " + "; ".join(failed))
        if not allow_missing:
            raise RuntimeError(msg + ". Pass allow_missing=True to score only the readable models.")
        log(f"  WARNING (allow_missing): {msg}")
    if not residuals:
        raise RuntimeError("no readable forecasts to evaluate")

    model_ids = list(residuals)
    log(f"Evaluating {len(model_ids)} model(s) over {len(grid.targets)} target(s), "
        f"{grid.n_h} horizon(s) on the common sample.")

    # ---- Phase 2: block-streamed scoring ----
    truth_stats: dict[tuple[str, tuple], dict] = defaultdict(_new_truth_entry)
    model_stats: dict[tuple[str, tuple], dict[str, dict]] = defaultdict(dict)
    n_common_total = 0
    targets_seen: set[str] = set()
    targets_with_common: set[str] = set()

    for target, horizon, sl, act, base in grid.iter_blocks():
        act_fin = ~np.isnan(act)
        if not act_fin.any():
            continue
        targets_seen.add(target)
        res_blocks = {m: np.asarray(residuals[m]["residual"][sl]) for m in model_ids}

        common = act_fin & np.isfinite(base)
        for r in res_blocks.values():
            common &= np.isfinite(r)
        if mask_bits is not None:
            common &= mask_bits[sl]
        n_c = int(common.sum())
        n_common_total += n_c
        if n_c > 0:
            targets_with_common.add(target)

        # model-independent truth stats (shared R^2 denominator)
        ac_common = (act - base)[common]
        sy = float(ac_common.sum())
        sy2 = float((ac_common ** 2).sum())
        q_present = np.flatnonzero(
            np.bincount(grid.quarter_codes[act_fin], minlength=grid.n_quarter_codes)).tolist()
        for level in AGG_LEVELS:
            gv = _group_vals(level, target, horizon)
            e = truth_stats[(level, gv)]
            e["quarters"].update(q_present)
            e["n_complete"] += n_c
            e["sum_y"] += sy
            e["sum_y2"] += sy2

        for m in model_ids:
            r = res_blocks[m]
            n_obs = int(np.count_nonzero(~np.isnan(r)))
            r_c = r[common].astype(np.float64)
            n_metric = int(np.isfinite(r_c).sum())
            ss_res = float((r_c ** 2).sum()) if n_metric > 0 else 0.0
            sae = float(np.abs(r_c).sum()) if n_metric > 0 else 0.0

            n_nll = n_crps = n_z2 = n_cover = 0
            sum_nll = sum_crps = sum_z2 = sum_cover = 0.0
            if m in sigma_family:
                fam, dfv = sigma_family[m]
                sig = np.asarray(residuals[m]["sigma"][sl], dtype=np.float64)
                valid = common & np.isfinite(sig) & (sig > 0.0)
                if valid.any():
                    res_v = r[valid].astype(np.float64)
                    sig_v = sig[valid]
                    nll_row = nll_per_entry(res_v, sig_v, fam, dfv)
                    good = np.isfinite(nll_row)
                    n_nll = int(good.sum())
                    sum_nll = float(nll_row[good].sum()) if n_nll else 0.0
                    crps_row = crps_per_entry(res_v, sig_v, fam, dfv)
                    good_c = np.isfinite(crps_row)
                    n_crps = int(good_c.sum())
                    sum_crps = float(crps_row[good_c].sum()) if n_crps else 0.0
                    if fam == "gaussian":
                        z2_row = (res_v / sig_v) ** 2
                        good_z = np.isfinite(z2_row)
                        n_z2 = int(good_z.sum())
                        sum_z2 = float(z2_row[good_z].sum()) if n_z2 else 0.0
                        cover_row = (np.abs(res_v / sig_v) <= _Z95).astype(np.float64)
                        n_cover = int(good_z.sum())
                        sum_cover = float(cover_row[good_z].sum()) if n_cover else 0.0

            for level in AGG_LEVELS:
                gv = _group_vals(level, target, horizon)
                acc = model_stats[(level, gv)].setdefault(m, _new_model_entry())
                acc["n_obs"] += n_obs
                acc["n_metric"] += n_metric
                acc["ss_res"] += ss_res
                acc["sae"] += sae
                acc["sum_nll"] += sum_nll
                acc["n_nll"] += n_nll
                acc["sum_crps"] += sum_crps
                acc["n_crps"] += n_crps
                acc["sum_z2"] += sum_z2
                acc["n_z2"] += n_z2
                acc["sum_cover"] += sum_cover
                acc["n_cover"] += n_cover

    dropped = sorted(targets_seen - targets_with_common)

    # ---- self-check: n_metric == n_complete per (target, horizon) ----
    invariant_ok = True
    for (level, gv), te in truth_stats.items():
        if level != "by_target_horizon":
            continue
        mb = model_stats.get((level, gv), {})
        for m in model_ids:
            nm = mb[m]["n_metric"] if m in mb else 0
            if nm != te["n_complete"]:
                invariant_ok = False
    if verbose:
        if invariant_ok:
            log("  Common-sample invariant holds (every model scored on the identical intersection).")
        else:
            log("  *** WARNING: common-sample invariant VIOLATED -- pooled metrics not trustworthy.")
        log(f"  Common metric sample: {n_common_total:,} cell(s) present-and-finite in all "
            f"{len(model_ids)} model(s), over {len(targets_with_common)} target(s).")
        if dropped:
            log(f"  NOTE: {len(dropped)} target(s) dropped from all levels (no all-models "
                f"common cell): {', '.join(dropped[:12])}{' ...' if len(dropped) > 12 else ''}")

    # ---- finalize ----
    metrics: dict[str, pd.DataFrame] = {}
    for level, keys in AGG_LEVELS.items():
        rows = []
        for (lv, gv), te in truth_stats.items():
            if lv != level:
                continue
            n = te["n_complete"]
            ss_total = (te["sum_y2"] - (te["sum_y"] ** 2) / n) if n > 0 else 0.0
            n_quarters = len(te["quarters"])
            mb = model_stats.get((lv, gv), {})
            for m in model_ids:
                acc = mb.get(m)
                n_obs = acc["n_obs"] if acc else 0
                avg_obs = n_obs / n_quarters if n_quarters > 0 else np.nan
                if n > 0 and acc is not None:
                    mae = acc["sae"] / n
                    mse = acc["ss_res"] / n
                    r2 = (1.0 - acc["ss_res"] / (ss_total + 1e-12)) if ss_total > 1e-15 else np.nan
                else:
                    mae = mse = r2 = np.nan
                nll = (acc["sum_nll"] / acc["n_nll"]) if (acc and acc["n_nll"] > 0) else np.nan
                crps = (acc["sum_crps"] / acc["n_crps"]) if (acc and acc["n_crps"] > 0) else np.nan
                z2 = (acc["sum_z2"] / acc["n_z2"]) if (acc and acc["n_z2"] > 0) else np.nan
                cover95 = (acc["sum_cover"] / acc["n_cover"]) if (acc and acc["n_cover"] > 0) else np.nan
                row = {"model": m, "r2": r2, "mae": mae, "mse": mse, "nll": nll,
                       "crps": crps, "z2": z2, "cover95": cover95, "avg_obs": avg_obs,
                       "n_complete": n}
                for k, v in zip(keys, gv):
                    row[k] = v
                rows.append(row)
        df = pd.DataFrame(rows)
        if "horizon" in df.columns and not df.empty:
            df["horizon"] = df["horizon"].astype(int)
        metrics[level] = df

    return EvalResult(metrics=metrics, n_common=n_common_total,
                      invariant_ok=invariant_ok, dropped_targets=dropped)


def _resolve_sample_mask(sample_mask, grid: TruthGrid, log) -> np.ndarray:
    """Return a length-``grid.n_cells`` bool mask from either a grid-aligned bit
    array (ndarray or ``.npy`` path) or a keys frame/parquet."""
    obj = sample_mask
    if isinstance(obj, (str, Path)) and str(obj).endswith(".npy"):
        obj = np.load(obj)
    if isinstance(obj, np.ndarray):
        n_cells = grid.n_cells
        packed_len = (n_cells + 7) // 8
        # Validate the source size BEFORE any truncation, so a mask built against a
        # DIFFERENT (e.g. larger) grid errors loudly instead of being silently
        # sliced to n_cells and applied to the wrong cells.
        if obj.dtype == np.uint8 and obj.size == packed_len:
            bits = np.unpackbits(obj)[:n_cells]           # canonical packbits form
        elif obj.size == n_cells:
            bits = obj                                    # already one entry per cell
        else:
            raise ValueError(
                f"grid-aligned sample mask has {obj.size} entries but this truth's grid "
                f"has {n_cells} cells (expected a length-{n_cells} bool array or its "
                f"{packed_len}-byte np.packbits); build the mask against the canonical "
                f"tabular_test (same firms/quarters/targets).")
        bits = bits.astype(bool)
        log(f"Sample mask (grid-aligned): {int(bits.sum()):,} of {n_cells:,} cells.")
        return bits
    # keys form
    mdf = pd.read_parquet(obj) if isinstance(obj, (str, Path)) else obj
    mrow_id, mvalid = grid.map_forecast_rows(normalize_columns(mdf))
    bits = np.zeros(grid.n_cells, dtype=bool)
    bits[mrow_id[mvalid]] = True
    log(f"Sample mask (keys): {int(bits.sum()):,} grid cell(s) "
        f"({int(mvalid.sum()):,} of {len(mdf):,} key rows on-grid).")
    return bits


def _iter_forecast_list(items):
    """Yield ``(name, obj)`` from a list of names / (name, df) / (name, path)."""
    for it in items:
        if isinstance(it, (tuple, list)) and len(it) == 2:
            yield it[0], it[1]
        else:
            yield str(it), it
