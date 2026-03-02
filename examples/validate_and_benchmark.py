"""End-to-end validation and benchmarking scorecard for the framework."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import supraharmonic_aggregation as sha
from scripts import config as feeder_config
from scripts.feeder_sim import _build_kernel_lookup, _interp_complex
from supraharmonic_aggregation.analysis.analytical import compute_analytical_statistics
from supraharmonic_aggregation.analysis.robustness import (
    run_multiseed_validation_study,
    summarize_multiseed_rows,
)
from supraharmonic_aggregation.benchmark.compare import compare_with_feeder_benchmark
from supraharmonic_aggregation.benchmark.independent import IndependentBenchmarkRunner
from supraharmonic_aggregation.config import AnalysisConfig


def _parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    values = [part.strip() for part in raw.split(",")]
    out: list[float] = []
    for value in values:
        if value:
            out.append(float(value))
    return out


def _parse_seeds(raw: str) -> list[int]:
    seeds = [part.strip() for part in raw.split(",")]
    out: list[int] = []
    for seed in seeds:
        if not seed:
            continue
        out.append(int(seed))
    if not out:
        raise ValueError("No valid seeds were provided.")
    return out


def _frequency_grid(start: float, stop: float, step: float) -> list[float]:
    n_steps = int(round((stop - start) / step))
    grid = [round(start + idx * step, 6) for idx in range(n_steps + 1)]
    if grid[-1] < stop:
        grid.append(stop)
    return grid


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


def _naive_baseline_rows(
    simulated: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    mean_rms = statistics.fmean(float(row["rms_abs_v"]) for row in simulated)
    mean_p95 = statistics.fmean(float(row["p95_abs_v"]) for row in simulated)
    return [
        {
            "frequency_khz": float(row["frequency_khz"]),
            "rms_abs_v": mean_rms,
            "p95_abs_v": mean_p95,
        }
        for row in simulated
    ]


def _summary_row(
    name: str,
    rows: list[dict[str, float | int | str]],
    rms_gate: float | None,
    p95_gate: float | None,
) -> dict[str, float | int | str | bool]:
    rms = [float(row["relative_error_rms"]) for row in rows]
    p95 = [float(row["relative_error_p95"]) for row in rows]
    rms_p90 = _quantile(rms, 0.90)
    p95_p90 = _quantile(p95, 0.90)
    pass_gate = True
    if rms_gate is not None:
        pass_gate = pass_gate and rms_p90 <= rms_gate
    if p95_gate is not None:
        pass_gate = pass_gate and p95_p90 <= p95_gate
    return {
        "benchmark_name": name,
        "n_frequencies": len(rows),
        "rms_error_mean": statistics.fmean(rms) if rms else 0.0,
        "rms_error_median": statistics.median(rms) if rms else 0.0,
        "rms_error_p90": rms_p90,
        "rms_error_max": max(rms) if rms else 0.0,
        "p95_error_mean": statistics.fmean(p95) if p95 else 0.0,
        "p95_error_median": statistics.median(p95) if p95 else 0.0,
        "p95_error_p90": p95_p90,
        "p95_error_max": max(p95) if p95 else 0.0,
        "passes_gate": pass_gate,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_scorecard(
    path: Path,
    config: AnalysisConfig,
    grid: list[float],
    seeds: list[int],
    samples: int,
    multiseed_samples: int,
    summaries: list[dict[str, float | int | str | bool]],
    multiseed_aggregate: dict[str, float | int],
) -> None:
    summary_by_name = {str(row["benchmark_name"]): row for row in summaries}
    internal = summary_by_name["framework_vs_internal"]
    independent = summary_by_name["framework_vs_independent"]
    naive = summary_by_name["naive_baseline_vs_internal"]

    framework_beats_naive = float(internal["rms_error_mean"]) < float(
        naive["rms_error_mean"]
    ) and float(internal["p95_error_mean"]) < float(naive["p95_error_mean"])
    multiseed_internal_pass = int(multiseed_aggregate["internal_pass_count"]) == int(
        multiseed_aggregate["n_seeds"]
    )
    multiseed_independent_pass = int(multiseed_aggregate["independent_pass_count"]) == int(
        multiseed_aggregate["n_seeds"]
    )
    overall_pass = (
        bool(internal["passes_gate"])
        and bool(independent["passes_gate"])
        and multiseed_internal_pass
        and multiseed_independent_pass
        and framework_beats_naive
    )

    lines = [
        "# Validation and Benchmark Scorecard",
        "",
        f"- Frequency grid: {grid[0]:.1f}-{grid[-1]:.1f} kHz, step {grid[1] - grid[0]:.1f} kHz, n={len(grid)}",
        f"- Internal/independent per-frequency samples: {samples}",
        f"- Multi-seed samples per frequency: {multiseed_samples}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Per-Benchmark Summary",
        "",
        "| Benchmark | RMS mean | RMS p90 | P95 mean | P95 p90 | Pass gate |",
        "|---|---:|---:|---:|---:|:---:|",
        (
            f"| framework_vs_internal | {float(internal['rms_error_mean']):.4f} | "
            f"{float(internal['rms_error_p90']):.4f} | {float(internal['p95_error_mean']):.4f} | "
            f"{float(internal['p95_error_p90']):.4f} | {'Y' if bool(internal['passes_gate']) else 'N'} |"
        ),
        (
            f"| framework_vs_independent | {float(independent['rms_error_mean']):.4f} | "
            f"{float(independent['rms_error_p90']):.4f} | {float(independent['p95_error_mean']):.4f} | "
            f"{float(independent['p95_error_p90']):.4f} | {'Y' if bool(independent['passes_gate']) else 'N'} |"
        ),
        (
            f"| naive_baseline_vs_internal | {float(naive['rms_error_mean']):.4f} | "
            f"{float(naive['rms_error_p90']):.4f} | {float(naive['p95_error_mean']):.4f} | "
            f"{float(naive['p95_error_p90']):.4f} | {'Y' if bool(naive['passes_gate']) else 'N'} |"
        ),
        "",
        "## Multi-Seed Robustness",
        "",
        f"- Internal pass count: {int(multiseed_aggregate['internal_pass_count'])}/{int(multiseed_aggregate['n_seeds'])}",
        f"- Independent pass count: {int(multiseed_aggregate['independent_pass_count'])}/{int(multiseed_aggregate['n_seeds'])}",
        f"- Internal RMS mean over seeds: {float(multiseed_aggregate['internal_rms_error_mean_over_seeds']):.4f}",
        f"- Independent RMS mean over seeds: {float(multiseed_aggregate['independent_rms_error_mean_over_seeds']):.4f}",
        "",
        "## Final Verdict",
        "",
        f"- Framework beats naive baseline: {'YES' if framework_beats_naive else 'NO'}",
        f"- Overall validation-and-benchmark gate: {'PASS' if overall_pass else 'FAIL'}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_density_values(
    synthetic_dir: str,
    override_values: str | None,
) -> list[float]:
    if override_values:
        vals = sorted(set(_parse_float_list(override_values)))
        if vals:
            return vals
    density_path = Path(synthetic_dir) / "synthetic_latest_density_sweep_statistics.csv"
    if density_path.exists():
        density_df = pd.read_csv(density_path)
        if "density" in density_df.columns:
            vals = sorted({float(v) for v in density_df["density"].dropna().tolist()})
            if vals:
                return vals
    return [6.0, 9.0, 12.0, 15.0, 18.0]


def _load_feeder_kernel_lookup(
    synthetic_dir: str,
) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    kernel_file = Path("feeder_benchmark_kernels.csv")
    if kernel_file.exists():
        kernels_df = pd.read_csv(kernel_file)
    else:
        fallback = Path(synthetic_dir) / "synthetic_latest_feeder_kernel_profiles.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                "Could not find feeder kernel data. Expected feeder_benchmark_kernels.csv "
                f"or {fallback}."
            )
        src = pd.read_csv(fallback)
        required = {
            "feeder_id",
            "feeder_name",
            "frequency_khz",
            "distance_m",
            "ztr_real_ohm",
            "ztr_imag_ohm",
        }
        missing = required - set(src.columns)
        if missing:
            raise ValueError(f"{fallback} missing required columns: {sorted(missing)}")
        kernels_df = src.rename(
            columns={
                "frequency_khz": "frequency_hz",
                "ztr_real_ohm": "Ztr_real_ohm",
                "ztr_imag_ohm": "Ztr_imag_ohm",
            }
        ).copy()
        kernels_df["frequency_hz"] = 1000.0 * kernels_df["frequency_hz"].astype(float)
        kernels_df["band_width_hz"] = 1000.0
    lookup = _build_kernel_lookup(kernels_df)
    names: dict[str, str] = {}
    for feeder_id, grp in kernels_df.groupby("feeder_id", sort=False):
        names[str(feeder_id)] = str(grp["feeder_name"].iloc[0])
    return lookup, names


def _integral_until(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    if x_max <= 0:
        return 0.0
    valid = x <= x_max + 1e-12
    xv = x[valid]
    yv = y[valid]
    if xv.size == 0:
        return 0.0
    if abs(xv[-1] - x_max) > 1e-9:
        y_end = float(np.interp(x_max, x, y))
        xv = np.append(xv, x_max)
        yv = np.append(yv, y_end)
    return float(np.trapezoid(yv, xv))


def _disc_second_moment_mean(
    d_grid: np.ndarray,
    z_abs_sq: np.ndarray,
    radius_m: float,
) -> float:
    if radius_m <= 0:
        return 0.0
    weighted = z_abs_sq * (2.0 * d_grid / (radius_m * radius_m))
    return _integral_until(d_grid, weighted, radius_m)


def _line_second_moment_mean(
    d_grid: np.ndarray,
    z_abs_sq: np.ndarray,
    line_length_m: float,
) -> float:
    if line_length_m <= 0:
        return 0.0
    integral = _integral_until(d_grid, z_abs_sq, line_length_m)
    return integral / line_length_m


def _best_line_length(
    d_grid: np.ndarray,
    z_abs_sq_by_freq: dict[float, np.ndarray],
    frequencies_khz: list[float],
    radius_m: float,
) -> tuple[float, float]:
    upper = min(float(np.max(d_grid)), float(radius_m))
    lower = min(1.0, upper)
    if upper <= lower:
        return max(upper, 1e-6), 0.0
    mu_disc = {
        freq: _disc_second_moment_mean(d_grid, z_abs_sq_by_freq[freq], radius_m)
        for freq in frequencies_khz
    }

    def objective(length: float) -> float:
        err = 0.0
        for freq in frequencies_khz:
            mu_b = _line_second_moment_mean(d_grid, z_abs_sq_by_freq[freq], length)
            denom = max(mu_disc[freq], 1e-12)
            rel = (mu_b - mu_disc[freq]) / denom
            err += rel * rel
        return err / max(len(frequencies_khz), 1)

    coarse = np.linspace(lower, upper, 200)
    coarse_obj = np.array([objective(float(length)) for length in coarse])
    idx = int(np.argmin(coarse_obj))
    best = float(coarse[idx])
    span = (upper - lower) / 200.0
    ref_low = max(lower, best - 2.0 * span)
    ref_high = min(upper, best + 2.0 * span)
    fine = np.linspace(ref_low, ref_high, 200)
    fine_obj = np.array([objective(float(length)) for length in fine])
    f_idx = int(np.argmin(fine_obj))
    return float(fine[f_idx]), float(fine_obj[f_idx])


def _sample_distances(
    geometry: str,
    region_radius_m: float,
    matched_length_m: float,
    branches: int,
    branch_lengths: np.ndarray,
    u_main: np.ndarray,
    u_aux: np.ndarray,
    trunk_offsets: np.ndarray | None = None,
    trunk_length_m: float | None = None,
    lateral_length_m: float | None = None,
) -> np.ndarray:
    if geometry == "disc":
        return region_radius_m * np.sqrt(u_main)
    if geometry == "star_branch":
        if branches <= 0:
            raise ValueError("branches must be positive for star_branch geometry.")
        lengths = (
            branch_lengths
            if branch_lengths.size
            else np.full(branches, matched_length_m, dtype=float)
        )
        probs = lengths / np.sum(lengths)
        cdf = np.cumsum(probs)
        which = np.searchsorted(cdf, u_aux, side="right")
        which = np.clip(which, 0, branches - 1)
        return lengths[which] * u_main
    if geometry == "two_level_branch":
        if branches <= 0:
            raise ValueError("branches must be positive for two_level_branch geometry.")
        if trunk_offsets is None or trunk_length_m is None or lateral_length_m is None:
            raise ValueError("Missing two-level geometry parameters.")
        probs = np.full(branches, 1.0 / branches, dtype=float)
        cdf = np.cumsum(probs)
        which = np.searchsorted(cdf, u_aux, side="right")
        which = np.clip(which, 0, branches - 1)
        return trunk_offsets[which] + lateral_length_m * u_main
    raise ValueError(f"Unsupported geometry: {geometry}")


def _quantile_np(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 100.0 * q))


def _run_geometry_ablation(
    output_dir: Path,
    frequencies_khz: list[float],
    samples: int,
    seeds: list[int],
    geometry: str,
    branches: int,
    branch_lengths: list[float],
    match_mode: str,
    synthetic_dir: str,
    density_values: str | None,
    region_radius_m: float,
    representative_density: float | None,
) -> dict[str, str]:
    kernel_lookup, feeder_names = _load_feeder_kernel_lookup(synthetic_dir=synthetic_dir)
    densities = _load_density_values(synthetic_dir=synthetic_dir, override_values=density_values)
    geometries = ["disc"] if geometry == "disc" else ["disc", geometry]
    branches = max(int(branches), 1)
    branch_lengths_arr = np.array(branch_lengths, dtype=float)
    if branch_lengths_arr.size not in (0, branches):
        raise ValueError("If --branch-lengths is provided, it must match --branches.")

    y_cache = {
        freq: 1j * 2.0 * math.pi * (freq * 1000.0) * feeder_config.Y_C_EQ_F
        for freq in frequencies_khz
    }
    match_rows: list[dict[str, float | str | int]] = []
    matched_lengths: dict[tuple[str, float], float] = {}

    for feeder_id in sorted(feeder_names):
        z_abs_sq_by_freq: dict[float, np.ndarray] = {}
        d_ref: np.ndarray | None = None
        for freq in frequencies_khz:
            key = (feeder_id, int(round(freq * 1000.0)))
            if key not in kernel_lookup:
                raise KeyError(f"Kernel lookup missing key {key}")
            d_grid, z_grid = kernel_lookup[key]
            if d_ref is None:
                d_ref = d_grid
            z_abs_sq_by_freq[freq] = np.abs(z_grid) ** 2
        if d_ref is None:
            continue
        single_l, objective = _best_line_length(
            d_grid=d_ref,
            z_abs_sq_by_freq=z_abs_sq_by_freq,
            frequencies_khz=frequencies_khz,
            radius_m=region_radius_m,
        )
        for density in densities:
            matched_lengths[(feeder_id, density)] = single_l
            match_rows.append(
                {
                    "feeder_id": feeder_id,
                    "feeder_name": feeder_names[feeder_id],
                    "density": density,
                    "match_mode": "second_moment_single_length"
                    if match_mode == "second_moment"
                    else "none",
                    "matched_length_m": single_l
                    if match_mode == "second_moment"
                    else region_radius_m,
                    "objective_mse": objective if match_mode == "second_moment" else 0.0,
                }
            )

    per_seed_rows: list[dict[str, float | str | int]] = []
    for feeder_idx, feeder_id in enumerate(sorted(feeder_names)):
        feeder_name = feeder_names[feeder_id]
        for density_idx, density in enumerate(densities):
            line_length = (
                matched_lengths[(feeder_id, density)]
                if match_mode == "second_moment"
                else region_radius_m
            )
            if geometry == "two_level_branch":
                trunk_length = max(0.35 * line_length, 1e-6)
                lateral_length = max((line_length - trunk_length) / 2.0, 1e-6)
            else:
                trunk_length = None
                lateral_length = None
            for seed in seeds:
                branch_rng = np.random.default_rng(
                    seed + 1_000_000 + feeder_idx * 10_000 + density_idx * 100
                )
                trunk_offsets = (
                    np.sort(branch_rng.uniform(0.0, max(trunk_length or 0.0, 1e-9), size=branches))
                    if geometry == "two_level_branch"
                    else None
                )
                for freq_idx, frequency in enumerate(frequencies_khz):
                    freq_hz = int(round(frequency * 1000.0))
                    d_grid, ztr_grid = kernel_lookup[(feeder_id, freq_hz)]
                    mean_n = density * math.pi * ((region_radius_m / 1000.0) ** 2)
                    seed_base = (
                        2_000_000
                        + 100_000 * feeder_idx
                        + 10_000 * density_idx
                        + 100 * seed
                        + freq_idx
                    )
                    rng = np.random.default_rng(seed_base)
                    n_vec = rng.poisson(mean_n, size=samples).astype(int)
                    total_n = int(n_vec.sum())
                    v_by_geometry: dict[str, np.ndarray] = {}
                    if total_n == 0:
                        zero = np.zeros(samples, dtype=float)
                        for geom_name in geometries:
                            v_by_geometry[geom_name] = zero
                    else:
                        idx = np.repeat(np.arange(samples, dtype=int), n_vec)
                        u_main = rng.random(total_n)
                        u_aux = rng.random(total_n)
                        phi = rng.uniform(0.0, 2.0 * math.pi, size=total_n)
                        i_sh = rng.lognormal(
                            mean=feeder_config.I_LOGN_MU_LN,
                            sigma=feeder_config.I_LOGN_SIGMA_LN,
                            size=total_n,
                        )
                        for geom_name in geometries:
                            distances = _sample_distances(
                                geometry=geom_name,
                                region_radius_m=region_radius_m,
                                matched_length_m=line_length,
                                branches=branches,
                                branch_lengths=branch_lengths_arr,
                                u_main=u_main,
                                u_aux=u_aux,
                                trunk_offsets=trunk_offsets,
                                trunk_length_m=trunk_length,
                                lateral_length_m=lateral_length,
                            )
                            ztr_d = _interp_complex(distances, d_grid, ztr_grid)
                            denom = 1.0 + y_cache[frequency] * ztr_d
                            contrib = ztr_d * i_sh * np.exp(1j * phi) / denom
                            v_real = np.bincount(idx, weights=contrib.real, minlength=samples)
                            v_imag = np.bincount(idx, weights=contrib.imag, minlength=samples)
                            v_by_geometry[geom_name] = np.abs(v_real + 1j * v_imag)

                    for geom_name, v_mag in v_by_geometry.items():
                        rms = float(np.sqrt(np.mean(v_mag * v_mag))) if v_mag.size else 0.0
                        p95 = _quantile_np(v_mag, 0.95)
                        p99 = _quantile_np(v_mag, 0.99)
                        per_seed_rows.append(
                            {
                                "feeder_id": feeder_id,
                                "feeder_name": feeder_name,
                                "density": density,
                                "frequency_khz": frequency,
                                "seed": seed,
                                "sample_size": samples,
                                "geometry": geom_name,
                                "rms_abs_v": rms,
                                "p95_abs_v": p95,
                                "p99_abs_v": p99,
                                "region_radius_m": region_radius_m,
                                "matched_length_m": line_length,
                                "branches": branches,
                            }
                        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    agg = per_seed_df.groupby(
        ["feeder_id", "feeder_name", "density", "frequency_khz", "geometry"], as_index=False
    )[["rms_abs_v", "p95_abs_v", "p99_abs_v"]].mean()

    comparison_rows: list[dict[str, float | str]] = []
    if geometry != "disc":
        for (feeder_id, density), grp in agg.groupby(["feeder_id", "density"], sort=True):
            disc = grp[grp["geometry"] == "disc"].set_index("frequency_khz")
            branch = grp[grp["geometry"] == geometry].set_index("frequency_khz")
            merged = disc[["rms_abs_v", "p95_abs_v", "p99_abs_v"]].join(
                branch[["rms_abs_v", "p95_abs_v", "p99_abs_v"]],
                how="inner",
                lsuffix="_disc",
                rsuffix=f"_{geometry}",
            )
            for freq, row in merged.iterrows():
                rms_rel = abs(
                    float(row[f"rms_abs_v_{geometry}"]) - float(row["rms_abs_v_disc"])
                ) / max(abs(float(row["rms_abs_v_disc"])), 1e-12)
                p95_rel = abs(
                    float(row[f"p95_abs_v_{geometry}"]) - float(row["p95_abs_v_disc"])
                ) / max(abs(float(row["p95_abs_v_disc"])), 1e-12)
                p99_rel = abs(
                    float(row[f"p99_abs_v_{geometry}"]) - float(row["p99_abs_v_disc"])
                ) / max(abs(float(row["p99_abs_v_disc"])), 1e-12)
                comparison_rows.append(
                    {
                        "feeder_id": feeder_id,
                        "feeder_name": str(grp["feeder_name"].iloc[0]),
                        "density": float(density),
                        "frequency_khz": float(freq),
                        "geometry": geometry,
                        "rms_abs_v_disc": float(row["rms_abs_v_disc"]),
                        f"rms_abs_v_{geometry}": float(row[f"rms_abs_v_{geometry}"]),
                        "p95_abs_v_disc": float(row["p95_abs_v_disc"]),
                        f"p95_abs_v_{geometry}": float(row[f"p95_abs_v_{geometry}"]),
                        "p99_abs_v_disc": float(row["p99_abs_v_disc"]),
                        f"p99_abs_v_{geometry}": float(row[f"p99_abs_v_{geometry}"]),
                        "rel_diff_rms": rms_rel,
                        "rel_diff_p95": p95_rel,
                        "rel_diff_p99": p99_rel,
                    }
                )
    comparison_df = pd.DataFrame(comparison_rows)
    if comparison_df.empty:
        comparison_df = pd.DataFrame(
            columns=[
                "feeder_id",
                "feeder_name",
                "density",
                "frequency_khz",
                "geometry",
                "rms_abs_v_disc",
                f"rms_abs_v_{geometry}",
                "p95_abs_v_disc",
                f"p95_abs_v_{geometry}",
                "p99_abs_v_disc",
                f"p99_abs_v_{geometry}",
                "rel_diff_rms",
                "rel_diff_p95",
                "rel_diff_p99",
            ]
        )

    summary_rows: list[dict[str, float | str]] = []
    if not comparison_df.empty:
        for (feeder_id, density, geom_name), grp in comparison_df.groupby(
            ["feeder_id", "density", "geometry"], sort=True
        ):
            grp_sorted = grp.sort_values("frequency_khz")
            metrics = {
                "rms": grp_sorted["rel_diff_rms"].to_numpy(dtype=float),
                "p95": grp_sorted["rel_diff_p95"].to_numpy(dtype=float),
                "p99": grp_sorted["rel_diff_p99"].to_numpy(dtype=float),
            }
            row: dict[str, float | str] = {
                "feeder_id": feeder_id,
                "feeder_name": str(grp_sorted["feeder_name"].iloc[0]),
                "density": float(density),
                "geometry": geom_name,
            }
            for name, values in metrics.items():
                row[f"{name}_rel_diff_mean"] = float(np.mean(values))
                row[f"{name}_rel_diff_p90"] = _quantile_np(values, 0.90)
                row[f"{name}_rel_diff_max"] = float(np.max(values))
                worst_idx = int(np.argmax(values))
                row[f"{name}_worst_frequency_khz"] = float(
                    grp_sorted.iloc[worst_idx]["frequency_khz"]
                )
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "feeder_id",
                "feeder_name",
                "density",
                "geometry",
                "rms_rel_diff_mean",
                "rms_rel_diff_p90",
                "rms_rel_diff_max",
                "rms_worst_frequency_khz",
                "p95_rel_diff_mean",
                "p95_rel_diff_p90",
                "p95_rel_diff_max",
                "p95_worst_frequency_khz",
                "p99_rel_diff_mean",
                "p99_rel_diff_p90",
                "p99_rel_diff_max",
                "p99_worst_frequency_khz",
            ]
        )

    representative = (
        representative_density
        if representative_density is not None
        else float(sorted(densities)[len(densities) // 2])
    )

    paths = {
        "geometry_per_seed": output_dir / "geometry_ablation_per_seed_metrics.csv",
        "geometry_per_frequency": output_dir / "geometry_ablation_per_frequency.csv",
        "table_g1": output_dir / "table_g1_geometry_ablation_summary.csv",
        "geometry_match": output_dir / "geometry_ablation_match_report.csv",
        "geometry_run_config": output_dir / "geometry_ablation_run_config.json",
    }
    per_seed_df.to_csv(paths["geometry_per_seed"], index=False)
    comparison_df.to_csv(paths["geometry_per_frequency"], index=False)
    summary_df.to_csv(paths["table_g1"], index=False)
    pd.DataFrame(match_rows).to_csv(paths["geometry_match"], index=False)
    paths["geometry_run_config"].write_text(
        json.dumps(
            {
                "frequencies_khz": frequencies_khz,
                "samples": samples,
                "seeds": seeds,
                "geometry_requested": geometry,
                "geometries_simulated": geometries,
                "density_values": densities,
                "region_radius_m": region_radius_m,
                "match_mode": match_mode,
                "branches": branches,
                "branch_lengths_m": branch_lengths if branch_lengths else None,
                "representative_density": representative,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate-and-benchmark",
        description="Run reviewer-grade validation and benchmark scorecard generation.",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_reports", help="Directory for report artifacts."
    )
    parser.add_argument(
        "--samples", type=int, default=None, help="Samples per frequency for main checks."
    )
    parser.add_argument(
        "--multiseed-samples",
        type=int,
        default=None,
        help="Samples per frequency for multi-seed checks.",
    )
    parser.add_argument(
        "--seeds", type=str, default="7,11,23,37,53", help="Comma-separated seed list."
    )
    parser.add_argument("--f-min", type=float, default=2.0, help="Minimum frequency in kHz.")
    parser.add_argument("--f-max", type=float, default=150.0, help="Maximum frequency in kHz.")
    parser.add_argument("--f-step", type=float, default=1.0, help="Frequency step in kHz.")
    parser.add_argument(
        "--geometry",
        type=str,
        default="disc",
        choices=["disc", "star_branch", "two_level_branch"],
        help="Geometry mode for geometry-ablation benchmark.",
    )
    parser.add_argument(
        "--branches", type=int, default=4, help="Number of branches for branched geometries."
    )
    parser.add_argument(
        "--branch-lengths",
        type=str,
        default=None,
        help="Optional comma-separated branch lengths in meters. Must match --branches when provided.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="second_moment",
        choices=["none", "second_moment"],
        help="Geometry matching mode for branched variants.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="synthetic_data",
        help="Directory containing synthetic helper assets (density sweep, feeder kernels).",
    )
    parser.add_argument(
        "--density-values",
        type=str,
        default=None,
        help="Optional comma-separated density values overriding synthetic density sweep.",
    )
    parser.add_argument(
        "--representative-density",
        type=float,
        default=None,
        help="Density value to highlight in Figure G1 generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    config = sha.default_config()
    frequencies = _frequency_grid(args.f_min, args.f_max, args.f_step)
    cfg = AnalysisConfig.from_dict({**config.to_dict(), "frequencies_khz": frequencies})

    samples = args.samples or max(cfg.review_ready_min_samples, cfg.monte_carlo_samples)
    multiseed_samples = args.multiseed_samples or max(samples // 2, 512)

    synthetic = sha.SyntheticDataGenerator(cfg, seed=cfg.seed).generate(
        n_samples=samples,
        include_complex=False,
        frequencies_khz=frequencies,
        include_measurement_noise=True,
    )

    analytical = compute_analytical_statistics(cfg)
    independent = IndependentBenchmarkRunner(cfg, seed=cfg.seed + 987_654).run(
        n_samples=samples,
        frequencies_khz=frequencies,
    )
    independent_validation = compare_with_feeder_benchmark(
        analytical=analytical,
        simulated=independent.statistics_frame,
    ).rows

    naive = _naive_baseline_rows(synthetic.statistics_frame)
    naive_validation = compare_with_feeder_benchmark(
        analytical=naive,
        simulated=synthetic.statistics_frame,
    ).rows

    multiseed_rows = run_multiseed_validation_study(
        config=cfg,
        seeds=seeds,
        n_samples=multiseed_samples,
        frequencies_khz=frequencies,
    )
    multiseed_aggregate = summarize_multiseed_rows(multiseed_rows)

    summary_rows = [
        _summary_row(
            "framework_vs_internal", synthetic.validation_frame, rms_gate=0.10, p95_gate=0.10
        ),
        _summary_row(
            "framework_vs_independent", independent_validation, rms_gate=0.20, p95_gate=0.20
        ),
        _summary_row("naive_baseline_vs_internal", naive_validation, rms_gate=None, p95_gate=None),
    ]

    per_frequency_rows: list[dict[str, float | int | str | bool]] = []
    for source_name, rows in (
        ("framework_vs_internal", synthetic.validation_frame),
        ("framework_vs_independent", independent_validation),
        ("naive_baseline_vs_internal", naive_validation),
    ):
        for row in rows:
            per_frequency_rows.append(
                {
                    "benchmark_name": source_name,
                    "frequency_khz": float(row["frequency_khz"]),
                    "relative_error_rms": float(row["relative_error_rms"]),
                    "relative_error_p95": float(row["relative_error_p95"]),
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_frequency": out_dir / "validation_benchmark_per_frequency.csv",
        "summary": out_dir / "validation_benchmark_summary.csv",
        "multiseed": out_dir / "validation_benchmark_multiseed.csv",
        "multiseed_aggregate": out_dir / "validation_benchmark_multiseed_aggregate.csv",
        "scorecard": out_dir / "validation_benchmark_scorecard.md",
        "run_config": out_dir / "validation_benchmark_run_config.json",
    }

    _write_csv(paths["per_frequency"], per_frequency_rows)
    _write_csv(paths["summary"], summary_rows)
    _write_csv(paths["multiseed"], multiseed_rows)
    _write_csv(paths["multiseed_aggregate"], [multiseed_aggregate])
    _write_scorecard(
        path=paths["scorecard"],
        config=cfg,
        grid=frequencies,
        seeds=seeds,
        samples=samples,
        multiseed_samples=multiseed_samples,
        summaries=summary_rows,
        multiseed_aggregate=multiseed_aggregate,
    )
    paths["run_config"].write_text(
        json.dumps(
            {
                "frequencies_khz": frequencies,
                "samples": samples,
                "multiseed_samples": multiseed_samples,
                "seeds": seeds,
                "gates": {
                    "framework_vs_internal": {"rms_p90_max": 0.10, "p95_p90_max": 0.10},
                    "framework_vs_independent": {"rms_p90_max": 0.20, "p95_p90_max": 0.20},
                },
                "config": cfg.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    geometry_paths = _run_geometry_ablation(
        output_dir=out_dir,
        frequencies_khz=frequencies,
        samples=samples,
        seeds=seeds,
        geometry=args.geometry,
        branches=args.branches,
        branch_lengths=_parse_float_list(args.branch_lengths),
        match_mode=args.match,
        synthetic_dir=args.synthetic_dir,
        density_values=args.density_values,
        region_radius_m=cfg.region_radius_m,
        representative_density=args.representative_density,
    )

    for path in paths.values():
        print(path)
    for path in geometry_paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
