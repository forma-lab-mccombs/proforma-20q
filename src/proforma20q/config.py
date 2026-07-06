"""Loaders for the bundled, version-controlled task configuration.

Everything the benchmark needs to define itself ships inside the package (see
``configs/``): the canonical task parameters, the pf_full feature set, and the
FF48 SIC ranges. None of it is WRDS-derived.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def canonical_reg_stats_path(tag: str = "r13_node_optionD_indfe_val8") -> Path:
    """Path to the bundled canonical ``regularization_stats`` artifact.

    Passing this (or ``--reg-stats canonical``) to the build pins the target /
    eval normalization to the published R13 space, so the scored targets are
    independent of the builder's Compustat vintage and BLAS/LAPACK stack.
    """
    fs = load_task_config()["benchmark"]["feature_set"]
    return REFERENCE_DIR / f"regularization_stats__{fs}__{tag}.parquet"


@lru_cache(maxsize=None)
def load_task_config() -> dict[str, Any]:
    """The canonical ProForma-20Q task definition (``configs/task.yaml``)."""
    with open(CONFIG_DIR / "task.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def load_feature_sets() -> dict[str, Any]:
    """The feature-set definitions (``configs/feature_sets.yaml``)."""
    with open(CONFIG_DIR / "feature_sets.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=None)
def load_ff48_ranges() -> dict[str, Any]:
    """The Fama-French 48 industry SIC ranges (``configs/ff48_sic_ranges.json``)."""
    with open(CONFIG_DIR / "ff48_sic_ranges.json", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_include_sets(name: str, sets: dict[str, Any], _seen: set[str]) -> list[str]:
    """Recursively expand a feature set's ``include_sets`` into a flat item list."""
    if name in _seen:
        return []
    _seen.add(name)
    spec = sets[name]
    items: list[str] = []
    for inc in spec.get("include_sets", []) or []:
        items.extend(_resolve_include_sets(inc, sets, _seen))
    for group in ("balance_sheet", "income_statement", "cash_flow", "computed"):
        items.extend(spec.get(group, []) or [])
    # Preserve first-seen order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            ordered.append(it)
    return ordered


def feature_set_items(feature_set: str = "pf_full") -> list[str]:
    """The ordered list of statement items (targets) for a feature set.

    For ``pf_full`` this is the canonical 78-item universe the benchmark
    forecasts jointly.
    """
    sets = load_feature_sets()["feature_sets"]
    if feature_set not in sets:
        raise KeyError(
            f"Unknown feature set {feature_set!r}. Available: {sorted(sets)}")
    return _resolve_include_sets(feature_set, sets, set())


def pf_full_targets() -> list[str]:
    """The 78 pf_full statement items -- the benchmark's joint forecast targets."""
    return feature_set_items("pf_full")
