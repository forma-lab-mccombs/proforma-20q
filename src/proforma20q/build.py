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

import gc
from dataclasses import dataclass, field
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


def required_raw_columns(feature_set: str | None = None, *,
                         include_attached: bool = True,
                         prefer_ytd_source: bool = False) -> list[str]:
    """Every raw column the build actually consumes, derived from the configs.

    ``comp.fundq`` ships 648 columns; the benchmark reads 82 of them. The list is
    computed, not hardcoded, so editing ``feature_sets.yaml`` /
    ``ytd_features.yaml`` moves the projection with it:

    * the feature set's statement items that exist natively in ``fundq``,
    * the inputs of every computed item in the feature set (``gpq`` needs
      ``revtq``/``cogsq``, ``fcfq`` needs ``oancfq``/``capxq``, ...),
    * the ``{base}y`` fiscal-YTD sources behind every ``{base}q`` item,
    * the scaling inputs, the Compustat keys, the fiscal-calendar fields YTD
      de-cumulation needs, and the descriptive metadata,
    * with ``include_attached``, the industry / CRSP-link columns that
      :func:`~proforma20q.download.download` attaches from other tables.

    ``prefer_ytd_source`` drops the ``{base}q`` form of any item that has a
    ``{base}y`` source -- ``convert_ytd_to_quarterly`` overwrites ``{base}q``
    wholesale from the YTD column, so the native one is dead weight. The
    download uses it (``comp.fundq`` has no ``oancfq`` to select in the first
    place); a parquet read does not, since a third-party panel may carry only
    the quarterly form.

    Ordering is stable (items in feature-set order, then keys) so a projected
    SQL SELECT and a projected parquet read agree.
    """
    task = load_task_config()
    feature_set = feature_set or task["benchmark"]["feature_set"]
    items = feature_set_items(feature_set)
    wanted = list(items)

    computed = _computed_definitions()
    for item in items:
        if item in computed:
            wanted.extend(computed[item][0])

    # A `{base}q` item that Compustat only reports fiscal-year-to-date is read as
    # `{base}y` and de-cumulated by convert_ytd_to_quarterly.
    ytd_bases = set(_ytd_bases())
    ytd_derived = {f"{b}q" for b in ytd_bases}
    for item in list(wanted):
        if item.endswith("q") and item[:-1] in ytd_bases:
            wanted.append(f"{item[:-1]}y")
    if prefer_ytd_source:
        wanted = [c for c in wanted if c not in ytd_derived]

    scaling = task.get("scaling", {})
    wanted.extend(scaling.get("features_to_sum", []) or [])
    wanted.extend(scaling.get("fallbacks", []) or [])

    # Keys, fiscal calendar (fqtr/fyearq drive YTD de-cumulation), metadata, and
    # the query filters -- kept so a downloaded panel can be re-audited without
    # a second pull.
    wanted.extend(["gvkey", "datadate", "fyearq", "fqtr", "tic", "conm",
                   "indfmt", "datafmt", "consol", "popsrc"])
    if include_attached:
        # not from fundq: sich/naicsh come from comp.co_industry, permno/permco
        # from the CCM link table. `sich` is load-bearing -- it drives both the
        # financial-sector exclusion and the FF48 industry map.
        wanted.extend(["sich", "naicsh", "permno", "permco"])

    seen: set[str] = set()
    out: list[str] = []
    for c in wanted:
        # computed items are formulas, never columns of the raw panel
        if c in computed or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


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
    calendar lags.

    Reference (single-firm) form of the expansion :func:`prepare_panel` performs
    for every firm at once; kept as the spec that vectorized expansion is tested
    against (``tests/test_build.py::test_vectorized_panel_matches_per_firm``).
    """
    per = g[qcol].dt.to_period("Q")
    full = pd.period_range(per.min(), per.max(), freq="Q")
    g = g.set_index(per)
    g = g.reindex(full)
    g[qcol] = full.to_timestamp(how="end")
    return g


# What `_quarter_ordinals` returns for NaT: pandas reports year == quarter == -1
# for the missing-period sentinel, so the arithmetic below yields -6.
_NAT_ORDINAL = -6


def _quarter_ordinals(values) -> np.ndarray:
    """Calendar-quarter ordinals (``year*4 + quarter - 1``) -- a dense integer
    axis on which a lag of L quarters is a shift of L positions."""
    per = pd.to_datetime(pd.Series(values)).dt.to_period("Q")
    return per.dt.year.to_numpy("int64") * 4 + per.dt.quarter.to_numpy("int64") - 1


@dataclass
class TabularPanel:
    """A gap-free per-firm quarterly panel held column-wise as flat arrays.

    Rows are ordered by (firm ascending, quarter ascending) and every firm
    occupies one contiguous block covering each calendar quarter between its
    first and last observation -- i.e. exactly what applying
    :func:`_contiguous_panel` to every firm and concatenating would produce, but
    built with vectorized index arithmetic and stored without any per-firm
    ``DataFrame``.

    Storing the panel as plain arrays (rather than a frame) is what lets
    :func:`build` release the raw WRDS panel before the wide matrix is allocated.
    """

    firm_ids: np.ndarray          # object, gvkey per row
    quarter: np.ndarray           # datetime64[ns] quarter-end per row
    qord: np.ndarray              # int64 calendar-quarter ordinal per row
    pos_in_firm: np.ndarray       # int32 rows since the firm's first quarter
    pos_from_end: np.ndarray      # int32 rows before the firm's last quarter
    values: dict[str, np.ndarray] = field(repr=False)   # item -> float64 column
    items: list[str] = field(default_factory=list)
    firm_col: str = "firm_id"
    qcol: str = "quarter"

    def __len__(self) -> int:
        return int(self.qord.shape[0])


def prepare_panel(raw_df: pd.DataFrame, items: list[str], *,
                  firm_col: str = "firm_id", qcol: str = "quarter") -> TabularPanel:
    """Expand the prepped raw panel onto the gap-free per-firm quarterly grid.

    Vectorized equivalent of ``groupby(firm).apply(_contiguous_panel)``: one
    ``np.repeat``-built index instead of 41,595 small frames.
    """
    present_items = list(dict.fromkeys(it for it in items if it in raw_df.columns))
    if "scale" not in raw_df.columns:
        raise KeyError("prepare_panel requires the 'scale' column (see compute_scale)")
    value_cols = list(dict.fromkeys(present_items + ["scale"]))

    codes, uniques = pd.factorize(raw_df[firm_col], sort=True)
    if (codes < 0).any():
        # factorize codes a missing key as -1, which then indexes the LAST firm's
        # block from the end and scatters the row into a different firm's panel.
        # The old per-firm groupby dropped these rows; silently mixing firms is
        # not an acceptable substitute.
        raise ValueError(
            f"{int((codes < 0).sum())} row(s) have a missing {firm_col}; drop them "
            "before building")
    qord_raw = _quarter_ordinals(raw_df[qcol])
    if (qord_raw == _NAT_ORDINAL).any():
        # NaT's sentinel year/quarter (-1, -1) produces a plausible-looking
        # ordinal ~8,000 quarters before the data, which inflates that firm's
        # contiguous range to ~8,100 rows before anything notices.
        raise ValueError(
            f"{int((qord_raw == _NAT_ORDINAL).sum())} row(s) have a missing {qcol}; "
            "drop them before building")
    order = np.lexsort((qord_raw, codes))          # (firm asc, quarter asc)
    codes = codes[order]
    qord_raw = qord_raw[order]

    n_firms = len(uniques)
    edges = np.searchsorted(codes, np.arange(n_firms + 1))
    first = qord_raw[edges[:-1]]
    last = qord_raw[edges[1:] - 1]
    lengths = last - first + 1
    total = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    panel_codes = np.repeat(np.arange(n_firms), lengths)
    pos_in_firm = (np.arange(total) - np.repeat(offsets, lengths)).astype(np.int32)
    pos_from_end = (np.repeat(lengths, lengths) - 1 - pos_in_firm).astype(np.int32)
    qord = np.repeat(first, lengths) + pos_in_firm

    # Where each observed row lands on the panel grid. Strictly increasing unless
    # the caller left duplicate (firm, quarter) keys -- which the per-firm
    # `reindex` form rejects outright, so reject them here too rather than
    # silently keeping whichever row happens to be written last.
    scatter = offsets[codes] + (qord_raw - first[codes])
    if total and np.any(np.diff(scatter) <= 0):
        raise ValueError(
            f"duplicate ({firm_col}, {qcol}) rows in the raw panel; de-duplicate "
            "before building (convert_ytd_to_quarterly already does this)")

    values: dict[str, np.ndarray] = {}
    for c in value_cols:
        col = raw_df[c].to_numpy(dtype=float)[order]
        arr = np.full(total, np.nan, dtype=np.float64)
        arr[scatter] = col
        values[c] = arr

    lo = int(qord.min())
    periods = pd.period_range(
        start=pd.Period(freq="Q", year=lo // 4, quarter=lo % 4 + 1),
        periods=int(qord.max()) - lo + 1, freq="Q")
    quarter = periods.to_timestamp(how="end").to_numpy()[qord - lo]
    firm_ids = np.asarray(uniques, dtype=object)[panel_codes]

    return TabularPanel(firm_ids=firm_ids, quarter=quarter, qord=qord,
                        pos_in_firm=pos_in_firm, pos_from_end=pos_from_end,
                        values=values, items=present_items,
                        firm_col=firm_col, qcol=qcol)


def _lag(vals: np.ndarray, lag: int, pos_in_firm: np.ndarray) -> np.ndarray:
    """Backward shift by ``lag`` within each firm (== ``groupby.shift(lag)``)."""
    if lag == 0:
        return vals
    out = np.full(vals.shape, np.nan)
    out[lag:] = vals[:-lag]
    out[pos_in_firm < lag] = np.nan     # would have reached into the previous firm
    return out


def _lead(vals: np.ndarray, lead: int, pos_from_end: np.ndarray) -> np.ndarray:
    """Forward shift by ``lead`` within each firm (== ``groupby.shift(-lead)``)."""
    out = np.full(vals.shape, np.nan)
    if lead:
        out[:-lead] = vals[lead:]
    else:
        out[:] = vals
    out[pos_from_end < lead] = np.nan   # would have reached into the next firm
    return out


def _stats_arrays(reg_stats: pd.DataFrame, q_lo: int, n_q: int) -> dict[str, tuple]:
    """feature -> (mu, sigma) as dense arrays indexed by ``qord - q_lo``."""
    rs = reg_stats.drop_duplicates(subset=["feature", "quarter"])
    pos = _quarter_ordinals(rs["quarter"]) - q_lo
    inside = (pos >= 0) & (pos < n_q)
    feat = rs["feature"].to_numpy()
    mu = rs["mu"].to_numpy(float)
    sig = rs["sigma"].to_numpy(float)
    out: dict[str, tuple] = {}
    for f in pd.unique(feat):
        sel = inside & (feat == f)
        a = np.full(n_q, np.nan)
        b = np.full(n_q, np.nan)
        a[pos[sel]] = mu[sel]
        b[pos[sel]] = sig[sel]
        out[f] = (a, b)
    return out


def _tabular_columns(present_items, target_items, recent, yoy, horizon,
                     *, with_scale: bool, dummy_ids) -> list[str]:
    """The wide matrix's column order (metadata keys excluded)."""
    names: list[str] = []
    if with_scale:
        names.append("scale_level_0")
    for item in present_items:
        names += [f"{item}_level_{lv}" for lv in range(recent)]
        names += [f"{item}_yoy_{yk}" for yk in range(yoy)]
        if item in target_items:
            names += [f"{item}_t{h}" for h in range(1, horizon + 1)]
    names += [f"indff48_{fid}" for fid in dummy_ids]
    return names


def _all_nan_rows(arr: np.ndarray, cols: np.ndarray, chunk: int = 32_768) -> np.ndarray:
    """Row-chunked ``isna().all(axis=1)`` over a column subset. Chunking keeps the
    boolean intermediate at a few hundred MB instead of ``n_rows x n_cols``."""
    out = np.empty(arr.shape[0], dtype=bool)
    if cols.size == 0:
        out[:] = True
        return out
    for a in range(0, arr.shape[0], chunk):
        b = min(a + chunk, arr.shape[0])
        out[a:b] = np.isnan(arr[a:b, cols]).all(axis=1)
    return out


def _fill_tabular(panel: TabularPanel, reg_stats: pd.DataFrame, target_items: list[str],
                  fe_cfg: dict, *, tabular_industry_fe: bool):
    """Allocate the wide float32 matrix once and fill it feature-by-feature.

    This is the memory-critical routine (port of the research repo's
    ``create_tabular_dataset``): a single contiguous ``(n_rows, n_cols)`` float32
    array is allocated up front and each feature writes its own columns in place,
    vectorized across all firms. The per-firm / per-column ``DataFrame``
    construction it replaces cost ~10 live copies of the output.

    The two cheap keep conditions (global-history quarter, non-NaN
    ``scale_level_0``) are evaluated on the panel *before* the matrix is
    allocated, so the ~30% of panel rows they drop -- gap-filled quarters and
    rows with no usable size proxy -- never cost 2,547 columns each.

    Returns ``(arr, names, col_idx, keep, sel)``: ``sel`` indexes the panel rows
    the matrix covers, and ``keep`` is the rest of the research repo's combined
    keep-mask (target -> feature) over those rows.
    """
    recent = fe_cfg["recent_levels"]
    yoy = fe_cfg["yoy_changes"]
    horizon = fe_cfg["forecast_horizon"]
    max_z = float(fe_cfg["max_abs_zscore"])
    min_lb = fe_cfg["min_lookback"]

    k_map = reg_stats.drop_duplicates("feature").set_index("feature")["k"].to_dict()
    q_lo = int(panel.qord.min())
    n_q = int(panel.qord.max()) - q_lo + 1
    qpos = panel.qord - q_lo
    stats = _stats_arrays(reg_stats, q_lo, n_q)
    _missing = (np.full(n_q, np.nan), np.full(n_q, np.nan))

    _ranges, unknown_id, id_to_name = _ff48_flat()
    dummy_ids = [fid for fid in sorted(id_to_name) if fid != unknown_id] \
        if tabular_industry_fe else []

    with_scale = "scale" in k_map
    names = _tabular_columns(panel.items, target_items, recent, yoy, horizon,
                             with_scale=with_scale, dummy_ids=dummy_ids)
    col_idx = {n: i for i, n in enumerate(names)}
    scale = panel.values["scale"]

    # -- keep conditions that do not need the wide matrix, applied first --
    # Condition 0: drop the earliest `min_lookback-1` GLOBAL quarters, which cannot
    # carry enough history after feature engineering. On the canonical R13 build
    # these quarters have no rows that survive the scale / all-NaN conditions, so
    # this is a no-op there, but it matches the reference for other sample windows.
    all_quarters = np.unique(panel.qord)
    if len(all_quarters) > min_lb:
        pre = panel.qord >= all_quarters[min_lb - 1]
    else:
        pre = np.ones(len(panel), dtype=bool)
    # Condition 1: scale_level_0 must be present (no divide-by-scale here; clamp +
    # float32 as the research repo).
    scale_lv0 = None
    if with_scale:
        mu_a, sg_a = stats.get("scale", _missing)
        scale_lv0 = _regularize(scale, scale, k_map["scale"], mu_a[qpos], sg_a[qpos],
                                max_z, divide_by_scale=False).astype(np.float32)
        pre &= ~np.isnan(scale_lv0)
    sel = np.flatnonzero(pre)
    del pre

    n_rows = sel.size
    arr = np.full((n_rows, len(names)), np.nan, dtype=np.float32)
    if with_scale:
        arr[:, col_idx["scale_level_0"]] = scale_lv0[sel]
        del scale_lv0

    max_lag = max(recent - 1, yoy - 1 + 4)
    for item in panel.items:
        k_i = k_map.get(item, 1.0)
        mu_a, sg_a = stats.get(item, _missing)
        mu_i, sg_i = mu_a[qpos], sg_a[qpos]
        raw_vals = panel.values[item]
        # Lags/leads are taken on the FULL panel and only then subset to `sel`:
        # a dropped row is still a legitimate lag source for a kept one.
        # UNCLIPPED normalized levels at lags 0..(recent+yoy+4); levels clamp on
        # output, YoY differences two UNCLIPPED normals THEN clamps (matching the
        # research repo, which never feeds a pre-clamped level into the YoY diff).
        norm_by_lag = [_normalize(_lag(raw_vals, lag, panel.pos_in_firm),
                                  scale, k_i, mu_i, sg_i)
                       for lag in range(max_lag + 1)]
        for lv in range(recent):
            arr[:, col_idx[f"{item}_level_{lv}"]] = np.clip(
                norm_by_lag[lv], -max_z, max_z)[sel]
        for yk in range(yoy):
            arr[:, col_idx[f"{item}_yoy_{yk}"]] = np.clip(
                norm_by_lag[yk] - norm_by_lag[yk + 4],
                -max_z * _YOY_Z_FACTOR, max_z * _YOY_Z_FACTOR)[sel]
        del norm_by_lag
        if item in target_items:
            for lead in range(1, horizon + 1):
                arr[:, col_idx[f"{item}_t{lead}"]] = _regularize(
                    _lead(raw_vals, lead, panel.pos_from_end),
                    scale, k_i, mu_i, sg_i, max_z)[sel]

    # -- remaining keep conditions (target -> feature), over the matrix rows --
    feat_cols = np.array([col_idx[c] for c in names
                          if ("_level_" in c or "_yoy_" in c) and c != "scale_level_0"],
                         dtype=np.intp)
    # Any recent level (0..recent-1), not just level_0 -- matches the research repo.
    tgt_level_cols = np.array([col_idx[f"{t}_level_{lv}"] for t in target_items
                               for lv in range(recent) if f"{t}_level_{lv}" in col_idx],
                              dtype=np.intp)
    keep = np.ones(n_rows, dtype=bool)
    if tgt_level_cols.size:
        keep &= ~_all_nan_rows(arr, tgt_level_cols)
    if feat_cols.size:
        keep &= ~_all_nan_rows(arr, feat_cols)
    if not keep.any():
        # Writing three empty parquets and logging success is the worst outcome
        # here; the usual causes are a `scale` that is NaN everywhere or frozen
        # reg-stats whose quarters do not overlap the panel.
        raise ValueError(
            f"no rows survive the keep-mask (panel {len(panel):,} rows, "
            f"{sel.size:,} with a usable scale). Check `scale` and that the "
            "regularization stats cover this panel's quarters.")
    return arr, names, col_idx, keep, sel


def _finish_tabular_block(arr, names, col_idx, firm_ids, qord, quarter, target_items,
                          fe_cfg, firm_ind, *, tabular_industry_fe, present_in_q0,
                          firm_col, qcol) -> pd.DataFrame:
    """Post-fill steps on an already-row-filtered block, in place on ``arr``.

    Mutating the array (not a wrapping frame) keeps every step allocation-free:
    ``.loc``-based writes on a 2,547-column frame are what a Copy-on-Write pandas
    would turn into a full copy of the block.
    """
    horizon = fe_cfg["forecast_horizon"]

    # -- present-in-q0 target masking --
    if present_in_q0:
        for t in target_items:
            l0 = col_idx.get(f"{t}_level_0")
            tcols = [col_idx[f"{t}_t{h}"] for h in range(1, horizon + 1)
                     if f"{t}_t{h}" in col_idx]
            if l0 is None or not tcols:
                continue
            absent = np.flatnonzero(np.isnan(arr[:, l0]))
            if not absent.size:
                continue
            if tcols[-1] - tcols[0] + 1 == len(tcols):
                # t1..t{horizon} are contiguous by construction -> slice, not fancy
                # index, so the write stays a strided memcpy.
                arr[absent, tcols[0]:tcols[-1] + 1] = np.nan
            else:
                arr[np.ix_(absent, np.asarray(tcols, dtype=np.intp))] = np.nan

    # -- imputation (per quarter) --
    # The research repo imputes EVERY ``_level_`` / ``_yoy_`` column, which
    # INCLUDES ``scale_level_0``. It is never itself NaN (kept rows require it),
    # so it is never filled -- but as an extra characteristic in the factor matrix
    # it shifts the loadings, hence the imputed values of every other cell. Match
    # that column set exactly or imputed cells diverge.
    impute_names = [c for c in names if "_level_" in c or "_yoy_" in c]
    impute_cols = np.array([col_idx[c] for c in impute_names], dtype=np.intp)
    if fe_cfg.get("imputation", {}).get("use") and impute_cols.size:
        n_factors = int(fe_cfg["imputation"].get("n_factors", 10))
        order = np.argsort(qord, kind="stable")
        bounds = np.flatnonzero(np.diff(qord[order])) + 1
        for rows in np.split(order, bounds):
            if rows.size < n_factors + 1:
                continue
            block = arr[np.ix_(rows, impute_cols)]
            if not np.isnan(block).any():
                continue
            # Fortran order deliberately: `impute_local_xs` runs an eigendecomposition
            # whose last-bit rounding depends on the input's memory layout, and a
            # column-extracted pandas block (what both the research repo and the
            # per-firm builder fed it) is column-major. Keeps imputed cells
            # bit-identical to those builders. Not theoretical: C-ordered input
            # moved 1,110 imputed `gdwlq` cells (max |delta| 8e-27, values whose
            # true magnitude is ~0) across 5 quarters of a 250-firm slice of the
            # real panel -- a degenerate factor problem where any perturbation
            # flips the result.
            filled = impute_local_xs(
                pd.DataFrame(np.asfortranarray(block), columns=impute_names),
                n_factors=n_factors)
            arr[np.ix_(rows, impute_cols)] = filled.to_numpy()

    # -- FF48 industry dummies --
    if tabular_industry_fe:
        _ranges, unknown_id, id_to_name = _ff48_flat()
        dummy_ids = [fid for fid in sorted(id_to_name) if fid != unknown_id]
        ind_ids = pd.Series(firm_ids).map(firm_ind).fillna(unknown_id).to_numpy("int64")
        first = col_idx[f"indff48_{dummy_ids[0]}"]
        arr[:, first:first + len(dummy_ids)] = (
            ind_ids[:, None] == np.asarray(dummy_ids)[None, :])

    df = pd.DataFrame(arr, columns=names, copy=False)
    df.insert(0, qcol, quarter)
    df.insert(0, firm_col, firm_ids)
    return df


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
    """Build the wide regularized tabular matrix (all firms, all quarters).

    Convenience form: expands the panel and returns the whole matrix as one
    frame. :func:`build` uses :func:`build_tabular_splits` instead, which shares
    this code path but never materializes the unsplit matrix.
    """
    panel = prepare_panel(raw_df, items, firm_col=firm_col, qcol=qcol)
    arr, names, col_idx, keep, sel = _fill_tabular(
        panel, reg_stats, target_items, fe_cfg, tabular_industry_fe=tabular_industry_fe)
    block = arr[keep]
    del arr
    gc.collect()
    rows = sel[keep]
    return _finish_tabular_block(
        block, names, col_idx, panel.firm_ids[rows], panel.qord[rows],
        panel.quarter[rows], target_items, fe_cfg, firm_ind,
        tabular_industry_fe=tabular_industry_fe, present_in_q0=present_in_q0,
        firm_col=firm_col, qcol=qcol)


def build_tabular_splits(
    panel: TabularPanel,
    reg_stats: pd.DataFrame,
    target_items: list[str],
    fe_cfg: dict,
    firm_ind: dict,
    splits: dict,
    *,
    tabular_industry_fe: bool = True,
    present_in_q0: bool = True,
):
    """Yield ``(split_name, frame)`` for train / val / test, one at a time.

    Identical output to ``split_tabular(build_tabular(...))`` -- every remaining
    step (present-in-q0 masking, per-quarter imputation, industry dummies,
    out-of-period masking) is row-local or per-quarter, and calendar splits never
    divide a quarter. Doing it per split means the unsplit ~12 GB matrix and its
    three split copies are never live at the same time.
    """
    train_end, val_end = splits["train_end_year"], splits["val_end_year"]
    arr, names, col_idx, keep, sel = _fill_tabular(
        panel, reg_stats, target_items, fe_cfg, tabular_industry_fe=tabular_industry_fe)
    year = panel.qord[sel] // 4
    for name, mask, end_year in (
        ("train", keep & (year <= train_end), train_end),
        ("val", keep & (year > train_end) & (year <= val_end), val_end),
        ("test", keep & (year > val_end), None),
    ):
        block = arr[mask]
        rows = sel[mask]
        # Out-of-period masking first: it only nulls ``_t`` cells, so it commutes
        # with present-in-q0 (also a ``_t`` NaN-setter) and with imputation (which
        # touches only ``_level_`` / ``_yoy_``). Doing it here keeps every write on
        # the bare array, before anything wraps it in a frame.
        if end_year is not None:
            _mask_out_of_period_inplace(block, col_idx, panel.qord[rows], end_year,
                                        target_items, fe_cfg["forecast_horizon"])
        df = _finish_tabular_block(
            block, names, col_idx, panel.firm_ids[rows], panel.qord[rows],
            panel.quarter[rows], target_items, fe_cfg, firm_ind,
            tabular_industry_fe=tabular_industry_fe, present_in_q0=present_in_q0,
            firm_col=panel.firm_col, qcol=panel.qcol)
        yield name, df
        del df, block
        gc.collect()


def _mask_out_of_period_inplace(arr, col_idx, qord, period_end_year, target_items, horizon):
    """Null targets whose landing quarter falls past ``period_end_year`` Q4
    (train/val only), so a split never carries a target from a later split."""
    boundary = pd.Period(f"{period_end_year}Q4", freq="Q")
    boundary_ord = boundary.year * 4 + boundary.quarter - 1
    for h in range(1, horizon + 1):
        cols = np.array([col_idx[f"{t}_t{h}"] for t in target_items
                         if f"{t}_t{h}" in col_idx], dtype=np.intp)
        if not cols.size:
            continue
        beyond = np.flatnonzero(qord + h > boundary_ord)
        if beyond.size:
            arr[np.ix_(beyond, cols)] = np.nan


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

    # Project to the columns the build consumes. A `SELECT f.*` panel carries 655
    # columns, of which 82 are read; loading all of them costs ~25 GB and is paid
    # on EVERY build, not just at download time.
    import fastparquet  # noqa: PLC0415 - engine is already a hard dependency
    available = list(fastparquet.ParquetFile(str(raw_path)).columns)
    # A panel may already carry the post-rename spellings (a third-party or
    # re-saved panel); ask for whichever form the file actually has, never both,
    # since renaming onto an existing column would duplicate it.
    renames = {"gvkey": "firm_id", "datadate": "quarter", "naicsh": "naics"}
    wanted_names = []
    for c in required_raw_columns():
        wanted_names.append(c)
        if c in renames and c not in available:
            wanted_names.append(renames[c])
    want = [c for c in dict.fromkeys(wanted_names) if c in available]
    log(f"Loading raw panel {raw_path} "
        f"({len(want)} of {len(available)} columns) ...")
    raw = pd.read_parquet(raw_path, engine="fastparquet", columns=want or None)
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
        # Expand onto the per-firm quarterly grid, then drop the raw panel BEFORE
        # the wide matrix is allocated -- the raw frame is several GB and nothing
        # downstream of here reads it.
        panel = prepare_panel(raw, items)
        del raw
        gc.collect()
        for name, part in build_tabular_splits(
                panel, reg_stats, target_items, fe, firm_ind, splits,
                tabular_industry_fe=task["industry"]["tabular_industry_fe"],
                present_in_q0=task["targets"]["present_in_q0"]):
            p = out_dir / f"tabular_{name}__{suffix}.parquet"
            part.to_parquet(p, engine="fastparquet", index=False)
            written[f"tabular_{name}"] = p
            log(f"  tabular_{name}: {len(part):,} rows, {part.shape[1]} cols")
            del part

    log(f"Build complete. suffix='{suffix}'")
    return written
