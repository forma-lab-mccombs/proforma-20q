"""The ProForma-20Q data build.

Turns the raw WRDS panel (``compustat_with_permno.parquet``) into the benchmark
artifacts. Faithful port of the Forma research repo's ``src/data`` pipeline;
every numeric constant, column name, and split rule matches the canonical R13
build so a rebuild reproduces the paper's data environment.

Two views are produced (ship both):

* **tabular** -- a wide lagged feature matrix (``tabular_{split}__{suffix}.parquet``).
  Columns per item: ``{item}_level_0..3`` (recent regularized levels),
  ``{item}_yoy_0..7`` (year-over-year changes in z-space), ``{item}_t1..t20``
  (regularized future targets); plus ``scale_level_0`` and ``indff48_0..47``
  industry dummies, and the metadata keys ``firm_id`` (gvkey string) /
  ``quarter`` (calendar quarter-end). The **tabular test file is also the eval
  ground truth** -- no separate truth file is written.

* **tuple** -- a sparse ``(firm_id, account_id, quarter, value, industry_id)``
  long form (``tuple_{split}__{suffix}.parquet``) holding RAW values; the
  regularization + |z|<=6 clamp is applied by the model's loader at runtime.

The regularization (all in ``transforms.py`` space):

    z = clip( ( asinh( k * (x / scale) ) - mu ) / sigma , -6, +6 )

where ``scale = |ltq| + |seqq| + 1e-3`` (row-wise fallback ``|atq| + 1e-3``),
``k`` is a per-item asinh constant estimated ON TRAIN ONLY, and ``mu`` / ``sigma``
are per-item, per-quarter trailing rolling-4 statistics (lookahead-free).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import eigh

from .config import CONFIG_DIR, feature_set_items, load_ff48_ranges, load_task_config
from .transforms import transform

# ---------------------------------------------------------------------------
# Metadata / naming
# ---------------------------------------------------------------------------
METADATA_COLS = ["firm_id", "quarter", "tic", "conm", "naics", "fyearq", "fqtr"]
BASE_QUARTER = pd.Period("1969Q4", freq="Q")  # tuple integer-quarter origin


def dataset_suffix(feature_set: str, dataset_tag: str | None) -> str:
    return f"{feature_set}__{dataset_tag}" if dataset_tag else feature_set


# ---------------------------------------------------------------------------
# FF48 industry mapping
# ---------------------------------------------------------------------------
def _ff48_flat() -> tuple[list[tuple[int, int, int]], int, dict[int, str]]:
    cfg = load_ff48_ranges()
    ranges: list[tuple[int, int, int]] = []
    id_to_name: dict[int, str] = {}
    for ind in cfg["industries"]:
        id_to_name[ind["id"]] = ind["name"]
        for lo, hi in ind["sic_ranges"]:
            ranges.append((int(lo), int(hi), int(ind["id"])))
    ranges.sort(key=lambda t: t[0])
    unknown_id = int(cfg.get("unknown_id", 48))
    return ranges, unknown_id, id_to_name


def sic_to_ff48(sich: pd.Series) -> pd.Series:
    """Map a SIC-code series to FF48 industry ids (first matching range wins;
    NaN / out-of-range -> unknown_id)."""
    ranges, unknown_id, _ = _ff48_flat()
    out = pd.Series(unknown_id, index=sich.index, dtype="int64")
    s = pd.to_numeric(sich, errors="coerce")
    for lo, hi, fid in ranges:
        m = s.between(lo, hi) & (out == unknown_id)
        out[m] = fid
    out[s.isna()] = unknown_id
    return out


def firm_ff48_map(raw_df: pd.DataFrame, firm_col: str = "firm_id") -> dict:
    """Firm -> FF48 id from each firm's MODAL ``sich``."""
    if "sich" not in raw_df.columns:
        return {}
    modal = raw_df.groupby(firm_col)["sich"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    ff = sic_to_ff48(modal)
    return dict(zip(modal.index, ff))


# ---------------------------------------------------------------------------
# YTD -> quarterly
# ---------------------------------------------------------------------------
def _ytd_bases() -> list[str]:
    with open(CONFIG_DIR / "ytd_features.yaml", encoding="utf-8") as fh:
        items = yaml.safe_load(fh)["ytd_features"]
    return [it[:-1] for it in items]  # strip trailing 'y'


def convert_ytd_to_quarterly(df: pd.DataFrame, firm_col: str = "firm_id") -> pd.DataFrame:
    """De-cumulate fiscal-year-to-date cash-flow items into quarterly flows.

    For YTD column ``{base}y`` write ``{base}q``: fqtr==1 keeps the YTD value;
    fqtr>1 subtracts the immediately-prior fiscal quarter's YTD within the same
    ``(firm, fyearq)`` (NaN if that prior quarter is absent). Source ``{base}y``
    columns are dropped.
    """
    df = df.copy()
    df = df.sort_values([firm_col, "fyearq", "fqtr", "quarter"])
    df = df.drop_duplicates(subset=[firm_col, "quarter"], keep="last")
    df = df.sort_values([firm_col, "quarter"])
    dropped = []
    for base in _ytd_bases():
        ytd = f"{base}y"
        if ytd not in df.columns:
            continue
        qcol = f"{base}q"
        df[qcol] = df[ytd]
        mask = df["fqtr"] > 1
        grp = df.groupby([firm_col, "fyearq"], sort=False)
        prev_ytd = grp[ytd].shift(1)
        prev_fqtr = grp["fqtr"].shift(1)
        prev_ytd = prev_ytd.where(prev_fqtr == (df["fqtr"] - 1))
        df.loc[mask, qcol] = df.loc[mask, ytd] - prev_ytd.loc[mask]
        dropped.append(ytd)
    return df.drop(columns=dropped)


# ---------------------------------------------------------------------------
# Computed items
# ---------------------------------------------------------------------------
def _computed_definitions() -> dict:
    """Formula table for computed items (requires-cols, fn). Ports
    feature_engineering.get_feature_definitions for the pf_full computed set and
    friends. Subtractive 'ex' items treat the subtracted NaN as 0."""
    def z(df, c):
        return df[c].fillna(0)
    return {
        "gpq": (["revtq", "cogsq"], lambda d: d["revtq"] - d["cogsq"]),
        "fcfq": (["oancfq", "capxq"], lambda d: d["oancfq"] - d["capxq"]),
        "dvcq": (["dvq", "dvpq"], lambda d: d["dvq"] - d["dvpq"]),
        "wcapq": (["actq", "lctq"], lambda d: d["actq"] - d["lctq"]),
        "aoq_ex_intanq": (["aoq", "intanq"], lambda d: d["aoq"] - z(d, "intanq")),
        "loq_ex_dr": (["loq", "drltq"], lambda d: d["loq"] - z(d, "drltq")),
        "xsgaq_ex_rd": (["xsgaq", "xrdq"], lambda d: d["xsgaq"] - z(d, "xrdq")),
        "neiq": (["sstkq", "prstkcq"], lambda d: z(d, "sstkq") - z(d, "prstkcq")),
    }


def add_computed_features(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """(Re)compute each computed item in ``items`` whose inputs are all present.

    Matches the research repo's ``compute_derived_features``: the formula
    OVERWRITES any like-named native Compustat column (e.g. ``wcapq``, ``gpq``,
    ``dvcq`` all ship in ``comp.fundq`` but the benchmark defines them as
    ``actq-lctq`` / ``revtq-cogsq`` / ``dvq-dvpq``). Skipping when the native
    column is present would silently diverge from the paper's targets.
    """
    df = df.copy()
    defs = _computed_definitions()
    for item in items:
        if item not in defs:
            continue
        req, fn = defs[item]
        if all(c in df.columns for c in req):
            df[item] = fn(df)
    return df


# ---------------------------------------------------------------------------
# Size proxy
# ---------------------------------------------------------------------------
def compute_scale(df: pd.DataFrame, scaling_cfg: dict) -> pd.Series:
    """``scale = sum|features_to_sum| + constant``, row-wise fallback
    ``|fallback| + constant``; residual non-positive/non-finite -> NaN."""
    constant = float(scaling_cfg.get("constant", 1e-3))
    to_sum = list(scaling_cfg.get("features_to_sum", []))
    fallbacks = list(scaling_cfg.get("fallbacks", []))

    scale = pd.Series(np.nan, index=df.index, dtype=float)
    if to_sum and all(f in df.columns for f in to_sum):
        scale = sum(df[f].abs() for f in to_sum) + constant

    def invalid(s):
        return s.isna() | ~np.isfinite(s) | (s <= 0)

    for fb in fallbacks:
        if fb not in df.columns:
            continue
        cur = invalid(scale)
        if not cur.any():
            break
        cand = df[fb].abs() + constant
        fill = cur & np.isfinite(cand) & (cand > 0)
        scale.loc[fill] = cand.loc[fill]

    scale = scale.replace([np.inf, -np.inf], np.nan)
    scale.loc[(scale <= 0) & scale.notna()] = np.nan
    return scale.astype(float)


# ---------------------------------------------------------------------------
# Regularization statistics (port of scaling.create_regularization_stats)
# ---------------------------------------------------------------------------
def calculate_k(vals: pd.Series) -> float:
    """k such that ``asinh(vals * k)`` has (pandas/Fisher) kurtosis nearest 3.0.
    Sweeps ``logspace(-2, 3, 250)``; defaults to 1.0."""
    best_k, best_diff = 1.0, float("inf")
    for k in np.logspace(-2, 3, num=250):
        kurt = transform(vals, k)
        kurt = pd.Series(kurt).kurtosis()
        if pd.isna(kurt):
            continue
        d = abs(kurt - 3.0)
        if d < best_diff:
            best_diff, best_k = d, float(k)
    return best_k


def _chain_pool(fdf, col, firm_col, qcol, scale_lag, feature_lags) -> pd.DataFrame:
    grouped = fdf.groupby(firm_col, sort=False)
    shifted_scale = fdf["scale"] if scale_lag == 0 else grouped["scale"].shift(scale_lag)
    safe_scale = shifted_scale.replace(0, np.nan)
    pools = []
    for L in feature_lags:
        x_lag = fdf[col] if L == 0 else grouped[col].shift(L)
        r = x_lag / safe_scale
        pools.append(pd.DataFrame({qcol: fdf[qcol].values, "r": r.values}))
    return pd.concat(pools, ignore_index=True).dropna(subset=["r"])


def create_regularization_stats(
    df: pd.DataFrame,
    items: list[str],
    fe_cfg: dict,
    train_cutoff: pd.Timestamp,
    firm_col: str = "firm_id",
    qcol: str = "quarter",
) -> pd.DataFrame:
    """Per (item, quarter) ``mu`` / ``sigma`` and per-item ``k``.

    ``k`` is train-only. ``mu`` / ``sigma`` are per-quarter pooled cross-sectional
    mean/std of ``asinh(k * r)`` (r = lagged value / lagged scale), then a
    trailing rolling-4-quarter mean. ``scale`` itself is special-cased.
    """
    mu_c = (fe_cfg["reg_mu_scale_lag"], list(fe_cfg["reg_mu_feature_lags"]))
    sig_c = (fe_cfg["reg_sig_scale_lag"], list(fe_cfg["reg_sig_feature_lags"]))
    k_c = (fe_cfg["reg_k_scale_lag"], list(fe_cfg["reg_k_feature_lags"]))

    df = df.copy()
    df[qcol] = pd.to_datetime(df[qcol])
    df_sorted = df.sort_values([firm_col, qcol])
    cols = [c for c in (items + ["scale"]) if c in df.columns]

    out = []
    for col in cols:
        if col == "scale":
            fdf = df_sorted[[firm_col, qcol, col]].dropna()
            if fdf.empty:
                continue
            train = fdf.loc[fdf[qcol] <= train_cutoff, col]
            k_val = calculate_k(train if not train.empty else fdf[col])
            fdf = fdf.assign(t=transform(fdf[col], k_val))
            qs = fdf.groupby(qcol)["t"].agg(["mean", "std"]).reset_index()
            qs = qs.rename(columns={"mean": "mu_raw", "std": "sigma_raw"})
        else:
            fdf = df_sorted[[firm_col, qcol, col, "scale"]].dropna()
            if fdf.empty:
                continue
            fdf = fdf.sort_values([firm_col, qcol])
            k_pool = _chain_pool(fdf, col, firm_col, qcol, *k_c)
            train_k = k_pool.loc[k_pool[qcol] <= train_cutoff, "r"]
            if train_k.empty:
                train_k = k_pool["r"]
            if train_k.empty:
                continue
            k_val = calculate_k(train_k)
            mu_pool = k_pool.copy() if mu_c == k_c else _chain_pool(fdf, col, firm_col, qcol, *mu_c)
            mu_pool["t"] = transform(mu_pool["r"], k_val)
            mu_q = mu_pool.groupby(qcol)["t"].mean().reset_index(name="mu_raw")
            if sig_c == mu_c:
                sig_q = mu_pool.groupby(qcol)["t"].std().reset_index(name="sigma_raw")
            else:
                sp = _chain_pool(fdf, col, firm_col, qcol, *sig_c)
                sp["t"] = transform(sp["r"], k_val)
                sig_q = sp.groupby(qcol)["t"].std().reset_index(name="sigma_raw")
            qs = pd.merge(mu_q, sig_q, on=qcol, how="inner")

        qs["mu_raw"] = qs["mu_raw"].fillna(0.0)
        qs["sigma_raw"] = qs["sigma_raw"].replace(0, np.nan).fillna(1.0) + 1e-8
        qs = qs.sort_values(qcol)
        qs["mu"] = qs["mu_raw"].rolling(window=4, min_periods=1).mean()
        qs["sigma"] = qs["sigma_raw"].rolling(window=4, min_periods=1).mean() + 1e-8
        sf = qs[[qcol, "mu", "sigma"]].copy()
        sf["feature"] = col
        sf["k"] = k_val
        out.append(sf)

    if not out:
        return pd.DataFrame(columns=["quarter", "mu", "sigma", "feature", "k"])
    alls = pd.concat(out, ignore_index=True).rename(columns={qcol: "quarter"})
    return alls


# ---------------------------------------------------------------------------
# Local-XS imputation (port of imputation.impute_local_xs)
# ---------------------------------------------------------------------------
def impute_local_xs(df: pd.DataFrame, n_factors: int = 10, gamma: float | None = None) -> pd.DataFrame:
    """Bryzgalova et al. (2024) Local-XS factor imputation of a (stocks x chars)
    frame. Observed values preserved; only NaNs filled. Deterministic."""
    import warnings

    data = df.to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nan_to_num(np.nanmean(data, axis=0), nan=0.0)
    centered = data - means
    N, L = centered.shape
    if gamma is None:
        gamma = 0.01 / L
    W = (~np.isnan(centered)).astype(float)
    X = np.nan_to_num(centered, nan=0.0)
    denom = W.T @ W
    denom[denom == 0] = 1.0
    Sigma = (X.T @ X) / denom
    evals, evecs = eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx][:n_factors], 1e-8)
    evecs = evecs[:, idx][:, :n_factors]
    Lambda = evecs @ np.diag(np.sqrt(evals))
    F = np.zeros((N, n_factors))
    I_K = np.eye(n_factors)
    for i in range(N):
        obs = W[i, :] == 1
        if not obs.any():
            continue
        Lo = Lambda[obs, :]
        yo = X[i, obs]
        lhs = (1.0 / L) * (Lo.T @ Lo) + gamma * I_K
        rhs = (1.0 / L) * (Lo.T @ yo)
        F[i, :] = np.linalg.solve(lhs, rhs)
    imputed = F @ Lambda.T + means
    # `data` already holds df.to_numpy(dtype=float, na_value=nan); copy it so the
    # destination is writable (pandas>=3.0 Copy-on-Write returns read-only arrays
    # from to_numpy) and fill only the originally-missing cells.
    out = data.copy()
    miss = np.isnan(data)
    out[miss] = imputed[miss]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------------
# Tabular construction
# ---------------------------------------------------------------------------
# YoY clamp widens the |z| bound by sqrt(2); the research repo uses the literal
# 1.4142 (not np.sqrt(2)), so match it for bit-exact clamp boundaries.
_YOY_Z_FACTOR = 1.4142


def _normalize(vals: np.ndarray, scale: np.ndarray, k: float, mu: np.ndarray,
               sigma: np.ndarray, divide_by_scale: bool = True) -> np.ndarray:
    """Regularized value BEFORE the |z| clamp: ``(asinh(k * v/scale) - mu)/sigma``.
    YoY changes are the difference of two of these UNCLIPPED normals (matching the
    research repo), so the clamp is applied once, after differencing."""
    working = vals / scale if divide_by_scale else vals
    return (np.arcsinh(working * k) - mu) / sigma


def _regularize(vals: np.ndarray, scale: np.ndarray, k: float, mu: np.ndarray,
                sigma: np.ndarray, max_abs_z: float, divide_by_scale: bool = True) -> np.ndarray:
    z = _normalize(vals, scale, k, mu, sigma, divide_by_scale=divide_by_scale)
    return np.clip(z, -max_abs_z, max_abs_z)


def _contiguous_panel(g: pd.DataFrame, qcol: str) -> pd.DataFrame:
    """Reindex one firm to a gap-free quarterly index so positional shifts equal
    calendar lags."""
    per = g[qcol].dt.to_period("Q")
    full = pd.period_range(per.min(), per.max(), freq="Q")
    g = g.set_index(per)
    g = g.reindex(full)
    g[qcol] = full.to_timestamp(how="end")
    return g


def build_tabular(
    raw_df: pd.DataFrame,
    reg_stats: pd.DataFrame,
    items: list[str],
    target_items: list[str],
    fe_cfg: dict,
    firm_ind: dict,
    *,
    tabular_industry_fe: bool = True,
    present_in_q0: bool = True,
    firm_col: str = "firm_id",
    qcol: str = "quarter",
) -> pd.DataFrame:
    """Build the wide regularized tabular matrix (all firms, all quarters)."""
    recent = fe_cfg["recent_levels"]
    yoy = fe_cfg["yoy_changes"]
    horizon = fe_cfg["forecast_horizon"]
    max_z = float(fe_cfg["max_abs_zscore"])
    min_lb = fe_cfg["min_lookback"]

    present_items = [it for it in items if it in raw_df.columns]
    k_map = reg_stats.drop_duplicates("feature").set_index("feature")["k"].to_dict()
    # (feature, quarter) -> (mu, sigma)
    mu_lut = {f: g.set_index("quarter")["mu"] for f, g in reg_stats.groupby("feature")}
    sig_lut = {f: g.set_index("quarter")["sigma"] for f, g in reg_stats.groupby("feature")}

    raw_df = raw_df.copy()
    raw_df[qcol] = pd.to_datetime(raw_df[qcol])

    panels = []
    for _fid, g in raw_df.sort_values([firm_col, qcol]).groupby(firm_col, sort=False):
        g = _contiguous_panel(g, qcol)
        n = len(g)
        scale = g["scale"].to_numpy(float)
        q_index = g[qcol]
        # per-row mu/sigma per item (mapped by the row's own quarter)
        out = {firm_col: _fid, qcol: g[qcol].to_numpy()}

        def stats_for(item):
            mu = q_index.map(mu_lut.get(item, pd.Series(dtype=float))).to_numpy(float)
            sg = q_index.map(sig_lut.get(item, pd.Series(dtype=float))).to_numpy(float)
            return k_map.get(item, 1.0), mu, sg

        # scale_level_0 (no divide-by-scale); clamp + float32 as the research repo.
        if "scale" in k_map:
            ks, mus, sgs = k_map["scale"], q_index.map(mu_lut["scale"]).to_numpy(float), \
                q_index.map(sig_lut["scale"]).to_numpy(float)
            out["scale_level_0"] = _regularize(
                scale, scale, ks, mus, sgs, max_z, divide_by_scale=False).astype(np.float32)

        for item in present_items:
            k_i, mu_i, sg_i = stats_for(item)
            raw_vals = g[item].to_numpy(float)
            # UNCLIPPED normalized levels at lags 0..(recent+yoy+4); levels clamp on
            # output, YoY differences two UNCLIPPED normals THEN clamps (matching the
            # research repo, which never feeds a pre-clamped level into the YoY diff).
            max_lag = max(recent - 1, yoy - 1 + 4)
            norm_by_lag = {}
            for lag in range(max_lag + 1):
                # backward shift by `lag` (== pandas groupby.shift(lag)): NaN-fill
                # the first `lag` rows, keep length n; all-NaN once lag >= n so a
                # short-history firm never trims from the wrong end.
                if lag == 0:
                    shifted = raw_vals
                else:
                    shifted = np.full(n, np.nan)
                    if lag < n:
                        shifted[lag:] = raw_vals[:n - lag]
                norm_by_lag[lag] = _normalize(shifted, scale, k_i, mu_i, sg_i)
            for lv in range(recent):
                out[f"{item}_level_{lv}"] = np.clip(norm_by_lag[lv], -max_z, max_z).astype(np.float32)
            for yk in range(yoy):
                out[f"{item}_yoy_{yk}"] = np.clip(
                    norm_by_lag[yk] - norm_by_lag[yk + 4],
                    -max_z * _YOY_Z_FACTOR, max_z * _YOY_Z_FACTOR).astype(np.float32)
            if item in target_items:
                for lead in range(1, horizon + 1):
                    # forward shift by `lead` (== pandas groupby.shift(-lead)):
                    # NaN-fill the last `lead` rows, keep length n; all-NaN once
                    # lead >= n.
                    fut = np.full(n, np.nan)
                    if lead < n:
                        fut[:n - lead] = raw_vals[lead:]
                    out[f"{item}_t{lead}"] = _regularize(
                        fut, scale, k_i, mu_i, sg_i, max_z).astype(np.float32)
        panels.append(pd.DataFrame(out))

    df = pd.concat(panels, ignore_index=True)

    # -- row filtering --
    feat_cols = [c for c in df.columns if ("_level_" in c or "_yoy_" in c) and c != "scale_level_0"]
    tgt_level_cols = [f"{t}_level_0" for t in target_items if f"{t}_level_0" in df.columns]
    keep = df["scale_level_0"].notna() if "scale_level_0" in df.columns else pd.Series(True, index=df.index)
    if tgt_level_cols:
        keep &= ~df[tgt_level_cols].isna().all(axis=1)
    if feat_cols:
        keep &= ~df[feat_cols].isna().all(axis=1)
    df = df[keep].reset_index(drop=True)

    # -- present-in-q0 target masking --
    if present_in_q0:
        for t in target_items:
            l0 = f"{t}_level_0"
            if l0 not in df.columns:
                continue
            absent = df[l0].isna()
            tcols = [f"{t}_t{h}" for h in range(1, horizon + 1) if f"{t}_t{h}" in df.columns]
            df.loc[absent, tcols] = np.nan

    # -- imputation (per quarter) --
    # The research repo imputes EVERY ``_level_`` / ``_yoy_`` column, which
    # INCLUDES ``scale_level_0``. It is never itself NaN (kept rows require it),
    # so it is never filled -- but as an extra characteristic in the factor matrix
    # it shifts the loadings, hence the imputed values of every other cell. Match
    # that column set exactly or imputed cells diverge.
    impute_cols = [c for c in df.columns if "_level_" in c or "_yoy_" in c]
    if fe_cfg.get("imputation", {}).get("use") and impute_cols:
        n_factors = int(fe_cfg["imputation"].get("n_factors", 10))
        parts = []
        for _q, gq in df.groupby(qcol, sort=False):
            block = gq[impute_cols]
            if block.isna().any().any() and len(gq) >= n_factors + 1:
                gq = gq.copy()
                gq[impute_cols] = impute_local_xs(block, n_factors=n_factors).astype("float32")
            parts.append(gq)
        df = pd.concat(parts).sort_index()

    # -- FF48 industry dummies --
    if tabular_industry_fe:
        _ranges, unknown_id, id_to_name = _ff48_flat()
        ind_ids = df[firm_col].map(firm_ind).fillna(unknown_id).astype("int64")
        # reference level (unknown_id) dropped; build all dummies at once to avoid
        # fragmenting the frame with 48 successive inserts.
        dummies = {
            f"indff48_{fid}": (ind_ids == fid).astype("float32")
            for fid in sorted(id_to_name) if fid != unknown_id
        }
        df = pd.concat([df, pd.DataFrame(dummies, index=df.index)], axis=1)

    return df


def _mask_out_of_period(df: pd.DataFrame, period_end_year: int, target_items: list[str],
                        horizon: int, qcol: str = "quarter") -> pd.DataFrame:
    """Null targets whose landing quarter falls past ``period_end_year`` Q4
    (train/val only), so a split never carries a target from a later split."""
    df = df.copy()
    boundary = pd.Period(f"{period_end_year}Q4", freq="Q")
    origin_q = pd.PeriodIndex(pd.to_datetime(df[qcol]).dt.to_period("Q"))
    for t in target_items:
        for h in range(1, horizon + 1):
            col = f"{t}_t{h}"
            if col not in df.columns:
                continue
            beyond = np.asarray((origin_q + h) > boundary)
            df.loc[beyond, col] = np.nan
    return df


def split_tabular(df, train_end_year, val_end_year, target_items, horizon, qcol="quarter"):
    """Calendar splits + out-of-period target masking on train/val."""
    yr = pd.to_datetime(df[qcol]).dt.year
    train = df[yr <= train_end_year]
    val = df[(yr >= train_end_year + 1) & (yr <= val_end_year)]
    test = df[yr >= val_end_year + 1]
    train = _mask_out_of_period(train, train_end_year, target_items, horizon, qcol)
    val = _mask_out_of_period(val, val_end_year, target_items, horizon, qcol)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tuple construction
# ---------------------------------------------------------------------------
def build_tuple(raw_df, account_cols, firm_ind, *, firm_col="firm_id", qcol="quarter",
                firm_id_map=None, account_id_map=None):
    """Melt raw values to ``(firm_id, account_id, quarter, value, industry_id)``.

    Integer id maps are built deterministically (sorted, from 0) when not
    supplied. Quarter is encoded as integer quarters since 1969Q4.
    """
    present = [c for c in account_cols if c in raw_df.columns]
    long = raw_df.melt(id_vars=[firm_col, qcol], value_vars=present,
                       var_name="account_name", value_name="value").dropna(subset=["value"])

    if firm_id_map is None:
        firm_id_map = {f: i for i, f in enumerate(sorted(raw_df[firm_col].unique()))}
    if account_id_map is None:
        account_id_map = {a: i for i, a in enumerate(sorted(present))}

    per = pd.to_datetime(long[qcol]).dt.to_period("Q")
    q_int = per.apply(lambda p: p.ordinal) - BASE_QUARTER.ordinal
    out = pd.DataFrame({
        "firm_id": long[firm_col].map(firm_id_map).astype("int64"),
        "account_id": long["account_name"].map(account_id_map).astype("int64"),
        "quarter": q_int.to_numpy(dtype="int64"),
        "value": long["value"].astype(float).to_numpy(),
    })
    _, unknown_id, _ = _ff48_flat()
    out["industry_id"] = long[firm_col].map(firm_ind).fillna(unknown_id).astype("int64").to_numpy()
    return out, firm_id_map, account_id_map


def _year_to_max_q(year: int) -> int:
    return (pd.Period(f"{year}Q4", freq="Q").ordinal - BASE_QUARTER.ordinal)


def split_tuple(df, train_end_year, val_end_year, pre_quarters):
    """Integer-quarter splits; val/test include ``pre_quarters`` of pre-boundary
    history (needed to forecast as-of the split boundary)."""
    tq = _year_to_max_q(train_end_year)
    vq = _year_to_max_q(val_end_year)
    train = df[df["quarter"] <= tq]
    val = df[(df["quarter"] > tq - pre_quarters) & (df["quarter"] <= vq)]
    test = df[df["quarter"] > vq - pre_quarters]
    return train, val, test


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build(
    raw_path,
    out_dir="data/processed",
    *,
    dataset_tag: str | None = None,
    which=("tabular", "tuple"),
    reg_stats=None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Build the ProForma-20Q artifacts from the raw WRDS panel.

    Args:
        raw_path: path to ``compustat_with_permno.parquet``.
        out_dir: output directory for the processed artifacts.
        dataset_tag: dataset tag (default from ``task.yaml`` -> the R13 tag).
        which: which views to build (``"tabular"``, ``"tuple"``).
        reg_stats: optional FROZEN regularization statistics to normalize against
            instead of re-estimating them from this panel -- a path to a
            ``regularization_stats__*.parquet`` or an in-memory frame with columns
            ``quarter, mu, sigma, feature, k``. Pinning the published canonical
            reg-stats makes the target/eval space independent of the builder's
            Compustat vintage and environment (see README / issue #1), so the
            paper's numbers reproduce even when a fresh vintage shifts the raw
            features. Default (None) re-estimates on the train split.
        verbose: progress printing.

    Returns:
        ``{artifact_name: path}``.
    """
    task = load_task_config()
    feature_set = task["benchmark"]["feature_set"]
    fe = task["feature_engineering"]  # imputation stays nested under fe["imputation"]
    splits = task["splits"]
    scaling_cfg = task["scaling"]
    dataset_tag = dataset_tag if dataset_tag is not None else "r13_node_optionD_indfe_val8"
    suffix = dataset_suffix(feature_set, dataset_tag)
    log = print if verbose else (lambda *a, **k: None)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading raw panel {raw_path} ...")
    raw = pd.read_parquet(raw_path, engine="fastparquet")
    raw = raw.rename(columns={"gvkey": "firm_id", "datadate": "quarter", "naicsh": "naics"})
    raw["firm_id"] = raw["firm_id"].astype(str)

    # sector exclusion (idempotent with download)
    for rng in task["universe"]["sector_exclusions"]["sic_ranges"]:
        if "sich" in raw.columns:
            raw = raw[raw["sich"].isna() | ~raw["sich"].between(rng["start"], rng["end"])]

    firm_ind = firm_ff48_map(raw)
    raw["quarter"] = pd.to_datetime(raw["quarter"]).dt.to_period("Q").dt.end_time

    raw = convert_ytd_to_quarterly(raw)
    items = feature_set_items(feature_set)
    raw = add_computed_features(raw, items)
    raw["scale"] = compute_scale(raw, scaling_cfg)

    target_items = [it for it in items if it in raw.columns]
    train_cutoff = pd.Timestamp(f"{splits['train_end_year']}-12-31")
    if reg_stats is None:
        log(f"Estimating regularization stats (train cutoff {train_cutoff.date()}) ...")
        reg_stats = create_regularization_stats(raw, items, fe, train_cutoff)
    else:
        if not isinstance(reg_stats, pd.DataFrame):
            log(f"Using FROZEN regularization stats from {reg_stats} ...")
            reg_stats = pd.read_parquet(reg_stats, engine="fastparquet")
        else:
            log("Using FROZEN regularization stats (in-memory) ...")
        reg_stats = reg_stats.copy()
        reg_stats["quarter"] = pd.to_datetime(reg_stats["quarter"])
    reg_path = out_dir / f"regularization_stats__{suffix}.parquet"
    reg_stats.to_parquet(reg_path, engine="fastparquet", index=False)

    written = {"reg_stats": reg_path}

    if "tuple" in which:
        log("Building tuple view ...")
        account_cols = [c for c in dict.fromkeys(items + target_items) if c in raw.columns] + \
            (["scale"] if "scale" in raw.columns else [])
        tup, _fm, _am = build_tuple(raw, account_cols, firm_ind)
        tr, va, te = split_tuple(tup, splits["train_end_year"], splits["val_end_year"],
                                 pre_quarters=fe["recent_levels"] + fe["yoy_changes"] - 1)
        for name, part in (("train", tr), ("val", va), ("test", te)):
            p = out_dir / f"tuple_{name}__{suffix}.parquet"
            part.to_parquet(p, engine="fastparquet")
            written[f"tuple_{name}"] = p
            log(f"  tuple_{name}: {len(part):,} rows")

    if "tabular" in which:
        log("Building tabular view ...")
        tab = build_tabular(raw, reg_stats, items, target_items, fe, firm_ind,
                            tabular_industry_fe=task["industry"]["tabular_industry_fe"],
                            present_in_q0=task["targets"]["present_in_q0"])
        tr, va, te = split_tabular(tab, splits["train_end_year"], splits["val_end_year"],
                                   target_items, fe["forecast_horizon"])
        for name, part in (("train", tr), ("val", va), ("test", te)):
            p = out_dir / f"tabular_{name}__{suffix}.parquet"
            part.to_parquet(p, engine="fastparquet", index=False)
            written[f"tabular_{name}"] = p
            log(f"  tabular_{name}: {len(part):,} rows, {part.shape[1]} cols")

    log(f"Build complete. suffix='{suffix}'")
    return written
