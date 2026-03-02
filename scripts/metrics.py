"""Metric calculations and uncertainty helpers."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("Percentile requested on empty array.")
    return float(np.percentile(values, q))


def compute_scalar_metrics(mags: np.ndarray, tau_v: float) -> dict[str, float]:
    if mags.size == 0:
        raise ValueError("compute_scalar_metrics received empty sample.")
    return {
        "mean": float(np.mean(mags)),
        "rms": float(math.sqrt(np.mean(mags * mags))),
        "p90": safe_percentile(mags, 90),
        "p95": safe_percentile(mags, 95),
        "p99": safe_percentile(mags, 99),
        "exceedance": float(np.mean(mags > tau_v)),
    }


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("bootstrap_ci received empty values.")
    if n_resamples <= 0:
        raise ValueError("bootstrap_ci requires n_resamples > 0.")
    rng = np.random.default_rng(seed)
    n = values.size
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(values[idx])
    alpha = 1.0 - ci_level
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def metrics_with_ci(
    mags: np.ndarray,
    tau_v: float,
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> dict[str, float]:
    m = compute_scalar_metrics(mags, tau_v=tau_v)

    # Fixed seed offsets for stable, process-independent reproducibility.
    seed_offsets: dict[str, int] = {
        "mean": 11,
        "rms": 23,
        "p95": 37,
        "p99": 41,
        "exceedance": 53,
    }
    fn_map: dict[str, Callable[[np.ndarray], float]] = {
        "mean": lambda x: float(np.mean(x)),
        "rms": lambda x: float(math.sqrt(np.mean(x * x))),
        "p95": lambda x: safe_percentile(x, 95),
        "p99": lambda x: safe_percentile(x, 99),
        "exceedance": lambda x: float(np.mean(x > tau_v)),
    }
    for name, fn in fn_map.items():
        lo, hi = bootstrap_ci(
            values=mags,
            stat_fn=fn,
            n_resamples=n_resamples,
            ci_level=ci_level,
            seed=seed + seed_offsets[name],
        )
        m[f"{name}_ci_lo"] = lo
        m[f"{name}_ci_hi"] = hi
    return m


def poisson_diagnostics_from_counts(counts: np.ndarray) -> dict[str, float]:
    if counts.size == 0:
        raise ValueError("poisson_diagnostics_from_counts received empty counts.")
    mean_n = float(np.mean(counts))
    var_n = float(np.var(counts))
    fano = var_n / mean_n if mean_n > 0 else float("nan")
    return {"mean_n": mean_n, "var_n": var_n, "fano": fano}


def cancellation_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    mean_re = float(df["Vagg_real_V"].mean())
    mean_im = float(df["Vagg_imag_V"].mean())
    mean_abs = float(df["Vagg_mag_V"].mean())
    return {
        "mean_re": mean_re,
        "mean_im": mean_im,
        "mean_abs": mean_abs,
        "abs_mean_re_ratio": abs(mean_re) / max(mean_abs, 1e-12),
        "abs_mean_im_ratio": abs(mean_im) / max(mean_abs, 1e-12),
    }


def denom_diagnostics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("denom_diagnostics received empty values.")
    return {
        "min": float(np.min(values)),
        "p01": safe_percentile(values, 1),
        "p05": safe_percentile(values, 5),
        "median": safe_percentile(values, 50),
    }
