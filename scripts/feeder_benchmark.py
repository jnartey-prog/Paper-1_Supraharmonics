"""Generate feeder-model benchmark kernels and extracted features."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from scripts import config


def _frequency_grid_hz() -> np.ndarray:
    return np.arange(
        config.FREQ_MIN_HZ,
        config.FREQ_MAX_HZ + 0.5 * config.FREQ_STEP_HZ,
        config.FREQ_STEP_HZ,
        dtype=float,
    )


def _distance_grid_m() -> np.ndarray:
    return np.arange(
        config.DIST_MIN_M,
        config.DIST_MAX_M + 0.5 * config.DIST_STEP_M,
        config.DIST_STEP_M,
        dtype=float,
    )


def _compute_ztr_matrix(
    feeder: config.FeederSpec,
    freqs_hz: np.ndarray,
    dists_m: np.ndarray,
) -> np.ndarray:
    def _line_input_impedance(
        gamma: complex, zc: complex, length_km: float, z_term: complex
    ) -> complex:
        if length_km <= 1e-12:
            return z_term
        t = np.tanh(gamma * length_km)
        denom = zc + z_term * t
        if abs(denom) <= 1e-18:
            return zc
        return zc * (z_term + zc * t) / denom

    out = np.zeros((len(freqs_hz), len(dists_m)), dtype=np.complex128)
    feeder_len_km = max(feeder.length_km, 1e-6)
    f_ref_hz = 2000.0

    for fi, f_hz in enumerate(freqs_hz):
        omega = 2.0 * np.pi * f_hz
        skin_mult = 1.0 + feeder.skin_effect_coeff * np.sqrt(max(f_hz, 1.0) / f_ref_hz)
        r_ac = feeder.R_ohm_per_km * skin_mult
        g_dielectric = omega * feeder.C_F_per_km * feeder.dielectric_tan_delta
        z = (r_ac + 1j * omega * feeder.L_H_per_km) / 1000.0
        y = (feeder.G_shunt_S + g_dielectric + 1j * omega * feeder.C_F_per_km) / 1000.0
        y_safe = y if abs(y) > 1e-18 else complex(1e-18, 0.0)

        gamma = np.sqrt(z * y_safe) + complex(1.0 / max(config.DIST_ATTENUATION_D0_M, 1e-6), 0.0)
        zc = np.sqrt(z / y_safe)
        z_source = complex(feeder.R_th_ohm, omega * feeder.L_th_H)
        z_right_term = zc

        for di, d_m in enumerate(dists_m):
            distance_km = max(float(d_m), 0.0) / 1000.0
            total_len_km = max(feeder_len_km, distance_km)
            left_len_km = min(distance_km, total_len_km)
            right_len_km = max(total_len_km - left_len_km, 0.0)

            zin_left = _line_input_impedance(gamma, zc, left_len_km, z_source)
            zin_right = _line_input_impedance(gamma, zc, right_len_km, z_right_term)
            zin_sum = zin_left + zin_right
            z_parallel = zin_left if abs(zin_sum) <= 1e-18 else (zin_left * zin_right) / zin_sum

            a = np.cosh(gamma * left_len_km)
            b = zc * np.sinh(gamma * left_len_km)
            transfer_den = a + (b / z_source)
            if abs(transfer_den) <= 1e-18:
                transfer_den = complex(1e-18, 0.0)
            out[fi, di] = z_parallel / transfer_den

    return out


def _extract_features(
    feeder: config.FeederSpec,
    freqs_hz: np.ndarray,
    dists_m: np.ndarray,
    ztr: np.ndarray,
) -> pd.DataFrame:
    zmag = np.abs(ztr)
    idx_0 = int(np.where(np.isclose(dists_m, 0.0))[0][0])
    idx_100 = int(np.where(np.isclose(dists_m, 100.0))[0][0])
    idx_500 = int(np.where(np.isclose(dists_m, 500.0))[0][0])
    idx_1000 = int(np.where(np.isclose(dists_m, 1000.0))[0][0])

    mask_fit = (dists_m >= 100.0) & (dists_m <= 1000.0)
    d_fit = dists_m[mask_fit]
    x = d_fit - d_fit.mean()
    x2 = float(np.sum(x * x))
    if x2 <= 0:
        raise ValueError("Distance fit grid is degenerate for fitted_alpha.")

    log_mag = np.log(np.maximum(zmag[:, mask_fit], 1e-16))
    slope = np.sum((log_mag - log_mag.mean(axis=1, keepdims=True)) * x[None, :], axis=1) / x2
    fitted_alpha = -slope

    # Resonance index per feeder from 0 m magnitude across frequency bins.
    z0 = zmag[:, idx_0]
    resonance_index = float(np.max(z0) / max(np.min(z0), 1e-16))

    df = pd.DataFrame(
        {
            "feeder_id": feeder.feeder_id,
            "feeder_name": feeder.feeder_name,
            "frequency_hz": freqs_hz,
            "band_width_hz": config.FREQ_STEP_HZ,
            "Zmag_0m": zmag[:, idx_0],
            "Zmag_100m": zmag[:, idx_100],
            "Zmag_500m": zmag[:, idx_500],
            "Zmag_1000m": zmag[:, idx_1000],
            "fitted_alpha": fitted_alpha,
            "resonance_index": resonance_index,
            "benchmark_label": "feeder-model benchmark",
        }
    )
    return df


def generate_feeder_benchmark(paths: config.Paths) -> dict[str, Any]:
    freqs_hz = _frequency_grid_hz()
    dists_m = _distance_grid_m()
    kernel_rows: list[pd.DataFrame] = []
    feature_rows: list[pd.DataFrame] = []
    spec_rows: list[dict[str, Any]] = []

    for feeder_id in sorted(config.FEEDER_SPECS.keys()):
        feeder = config.FEEDER_SPECS[feeder_id]
        ztr = _compute_ztr_matrix(feeder, freqs_hz=freqs_hz, dists_m=dists_m)
        re = ztr.real
        im = ztr.imag
        mag = np.abs(ztr)

        nf = len(freqs_hz)
        nd = len(dists_m)
        kernel_rows.append(
            pd.DataFrame(
                {
                    "feeder_id": feeder.feeder_id,
                    "feeder_name": feeder.feeder_name,
                    "frequency_hz": np.repeat(freqs_hz, nd),
                    "band_width_hz": config.FREQ_STEP_HZ,
                    "distance_m": np.tile(dists_m, nf),
                    "Ztr_real_ohm": re.reshape(-1),
                    "Ztr_imag_ohm": im.reshape(-1),
                    "Ztr_mag_ohm": mag.reshape(-1),
                    "benchmark_label": "feeder-model benchmark",
                }
            )
        )
        feature_rows.append(_extract_features(feeder, freqs_hz=freqs_hz, dists_m=dists_m, ztr=ztr))
        spec_rows.append(asdict(feeder))

    kernels_df = pd.concat(kernel_rows, ignore_index=True)
    features_df = pd.concat(feature_rows, ignore_index=True)
    specs_df = pd.DataFrame(spec_rows)

    paths.feeder_benchmark_kernels.parent.mkdir(parents=True, exist_ok=True)
    kernels_df.to_csv(paths.feeder_benchmark_kernels, index=False)
    features_df.to_csv(paths.feeder_benchmark_features, index=False)

    return {
        "kernels_path": paths.feeder_benchmark_kernels,
        "features_path": paths.feeder_benchmark_features,
        "kernels_df": kernels_df,
        "features_df": features_df,
        "specs_df": specs_df,
    }
