"""Feeder-model PCC simulations for benchmark validation."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from scripts import config, metrics


def _y_admittance_s(frequency_hz: float) -> complex:
    return 1j * 2.0 * np.pi * frequency_hz * config.Y_C_EQ_F


def _build_kernel_lookup(
    kernels_df: pd.DataFrame,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    grouped = kernels_df.groupby(["feeder_id", "frequency_hz"], sort=False)
    for (feeder_id, freq_hz), grp in grouped:
        g = grp.sort_values("distance_m")
        d = g["distance_m"].to_numpy(dtype=float)
        z = g["Ztr_real_ohm"].to_numpy(dtype=float) + 1j * g["Ztr_imag_ohm"].to_numpy(dtype=float)
        lookup[(str(feeder_id), int(round(float(freq_hz))))] = (d, z)
    return lookup


def _interp_complex(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    real = np.interp(x, xp, fp.real)
    imag = np.interp(x, xp, fp.imag)
    return real + 1j * imag


def _base_seed(feeder_id: str, setting_id: int, seed_setting: int) -> int:
    feeder_offset = {"A": 100_000, "B": 200_000, "C": 300_000}.get(feeder_id, 400_000)
    return int(config.RNG_SEED + feeder_offset + 97 * int(setting_id) + 13 * int(seed_setting))


def _z_value(ci_level: float) -> float:
    alpha = 1.0 - ci_level
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def generate_feeder_model_simulations(
    paths: config.Paths,
    baseline_scenario: pd.DataFrame,
    kernels_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if kernels_df is None:
        if not paths.feeder_benchmark_kernels.exists():
            raise FileNotFoundError(
                f"Required feeder kernels file not found: {paths.feeder_benchmark_kernels}"
            )
        kernels_df = pd.read_csv(paths.feeder_benchmark_kernels)

    required_cols = {
        "feeder_id",
        "feeder_name",
        "frequency_hz",
        "band_width_hz",
        "distance_m",
        "Ztr_real_ohm",
        "Ztr_imag_ohm",
    }
    missing = required_cols - set(kernels_df.columns)
    if missing:
        raise ValueError(f"feeder_benchmark_kernels.csv missing columns: {sorted(missing)}")

    kernel_lookup = _build_kernel_lookup(kernels_df)
    K = int(config.FEEDER_SIM_REALIZATIONS_PER_SETTING)
    z = _z_value(config.CI_LEVEL)

    paths.feeder_model_realizations.parent.mkdir(parents=True, exist_ok=True)
    if paths.feeder_model_realizations.exists():
        paths.feeder_model_realizations.unlink()

    write_header = True
    summary_rows: list[dict[str, Any]] = []
    n_realization_rows = 0

    for feeder_id in sorted(config.FEEDER_SPECS.keys()):
        feeder = config.FEEDER_SPECS[feeder_id]
        for row in baseline_scenario.itertuples(index=False):
            setting_id = int(row.setting_id)
            frequency_hz = float(row.frequency_hz)
            freq_key = int(round(frequency_hz))
            lambda_per_m2 = float(row.lambda_per_m2)
            region_R_m = float(row.region_R_m)
            tau_v = float(row.tau_V)
            band_width_hz = float(row.band_width_hz)
            seed_setting = int(getattr(row, "seed_setting", setting_id))

            key = (feeder_id, freq_key)
            if key not in kernel_lookup:
                raise KeyError(f"Missing feeder kernel for key={key}")
            dist_grid, ztr_grid = kernel_lookup[key]
            if region_R_m > float(np.max(dist_grid)):
                raise ValueError(
                    f"Scenario radius {region_R_m} exceeds feeder kernel distance max {np.max(dist_grid)}."
                )

            y_f = _y_admittance_s(frequency_hz)
            denom_grid = np.abs(1.0 + y_f * ztr_grid[dist_grid <= region_R_m + 1e-12])
            denom_mag_min = float(np.min(denom_grid))

            seed_base = _base_seed(
                feeder_id=feeder_id, setting_id=setting_id, seed_setting=seed_setting
            )
            rng = np.random.default_rng(seed_base)

            mean_n = lambda_per_m2 * np.pi * (region_R_m**2)
            n_vec = rng.poisson(mean_n, size=K).astype(int)
            total_n = int(n_vec.sum())
            v_agg = np.zeros(K, dtype=np.complex128)

            if total_n > 0:
                idx = np.repeat(np.arange(K, dtype=int), n_vec)
                d = region_R_m * np.sqrt(rng.random(total_n))
                phi = rng.uniform(0.0, 2.0 * np.pi, size=total_n)
                i_sh = rng.lognormal(
                    mean=config.I_LOGN_MU_LN, sigma=config.I_LOGN_SIGMA_LN, size=total_n
                )
                ztr_d = _interp_complex(d, dist_grid, ztr_grid)
                denom = 1.0 + y_f * ztr_d
                contrib = ztr_d * i_sh * np.exp(1j * phi) / denom
                v_real = np.bincount(idx, weights=contrib.real, minlength=K)
                v_imag = np.bincount(idx, weights=contrib.imag, minlength=K)
                v_agg = v_real + 1j * v_imag

            v_mag = np.abs(v_agg)
            seeds = seed_base + np.arange(K, dtype=int)
            real_df = pd.DataFrame(
                {
                    "feeder_id": feeder.feeder_id,
                    "feeder_name": feeder.feeder_name,
                    "setting_id": setting_id,
                    "frequency_hz": frequency_hz,
                    "band_width_hz": band_width_hz,
                    "lambda_per_m2": lambda_per_m2,
                    "region_R_m": region_R_m,
                    "realization_in_setting": np.arange(1, K + 1, dtype=int),
                    "seed": seeds,
                    "N": n_vec,
                    "Vagg_real_V": v_agg.real,
                    "Vagg_imag_V": v_agg.imag,
                    "Vagg_mag_V": v_mag,
                    "denom_mag_min_over_domain": denom_mag_min,
                    "benchmark_label": "feeder-model benchmark",
                }
            )
            real_df.to_csv(
                paths.feeder_model_realizations,
                mode="w" if write_header else "a",
                index=False,
                header=write_header,
            )
            write_header = False
            n_realization_rows += int(real_df.shape[0])

            m = metrics.compute_scalar_metrics(v_mag.astype(float), tau_v=tau_v)
            mean_v = float(np.mean(v_mag))
            std_v = float(np.std(v_mag, ddof=1))
            se_mean = std_v / max(np.sqrt(K), 1e-12)
            mean_ci_lo = mean_v - z * se_mean
            mean_ci_hi = mean_v + z * se_mean

            # Delta-method CI for RMS = sqrt(E[V^2]).
            y = v_mag * v_mag
            mean_y = float(np.mean(y))
            var_y = float(np.var(y, ddof=1))
            se_mean_y = np.sqrt(var_y / max(K, 1))
            rms_v = float(np.sqrt(max(mean_y, 0.0)))
            se_rms = float(se_mean_y / max(2.0 * rms_v, 1e-12))
            rms_ci_lo = max(0.0, rms_v - z * se_rms)
            rms_ci_hi = rms_v + z * se_rms

            # Approximate quantile CI using uncertainty of quantile probability.
            p_q = 0.99
            se_pq = float(np.sqrt(p_q * (1.0 - p_q) / max(K, 1)))
            q_lo = max(0.0, min(1.0, p_q - z * se_pq))
            q_hi = max(0.0, min(1.0, p_q + z * se_pq))
            p99_ci_lo = float(np.percentile(v_mag, 100.0 * q_lo))
            p99_ci_hi = float(np.percentile(v_mag, 100.0 * q_hi))

            p_exc = float(np.mean(v_mag > tau_v))
            se_exc = float(np.sqrt(max(p_exc * (1.0 - p_exc), 0.0) / max(K, 1)))
            exceed_ci_lo = max(0.0, p_exc - z * se_exc)
            exceed_ci_hi = min(1.0, p_exc + z * se_exc)

            summary_rows.append(
                {
                    "feeder_id": feeder.feeder_id,
                    "feeder_name": feeder.feeder_name,
                    "setting_id": setting_id,
                    "frequency_hz": frequency_hz,
                    "band_width_hz": band_width_hz,
                    "lambda_per_m2": lambda_per_m2,
                    "region_R_m": region_R_m,
                    "n_realizations": K,
                    "tau_V": tau_v,
                    "mean_Vmag": mean_v,
                    "mean_ci_lo": mean_ci_lo,
                    "mean_ci_hi": mean_ci_hi,
                    "rms_Vmag": rms_v,
                    "rms_ci_lo": rms_ci_lo,
                    "rms_ci_hi": rms_ci_hi,
                    "var_Vmag": float(np.var(v_mag)),
                    "p90_Vmag": m["p90"],
                    "p95_Vmag": m["p95"],
                    "p99_Vmag": m["p99"],
                    "p99_ci_lo": p99_ci_lo,
                    "p99_ci_hi": p99_ci_hi,
                    "exceed_tau": p_exc,
                    "exceed_ci_lo": exceed_ci_lo,
                    "exceed_ci_hi": exceed_ci_hi,
                    "mean_N": float(np.mean(n_vec)),
                    "var_N": float(np.var(n_vec)),
                    "denom_mag_min_over_domain": denom_mag_min,
                    "seed_setting": seed_base,
                    "benchmark_label": "feeder-model benchmark",
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["feeder_id", "frequency_hz", "lambda_per_m2", "region_R_m"]
    )
    summary_df.to_csv(paths.feeder_model_setting_summary, index=False)

    return {
        "realizations_path": paths.feeder_model_realizations,
        "setting_summary_path": paths.feeder_model_setting_summary,
        "summary_df": summary_df,
        "n_realizations_rows": n_realization_rows,
    }
