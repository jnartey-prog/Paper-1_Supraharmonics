"""Deterministic moment-based analytical statistics."""

from __future__ import annotations

import math

from ..config import AnalysisConfig
from ..core.kernel import ExponentialKernel
from ..models import StatisticsFrame
from .tail import adaptive_threshold

_Z95 = 1.6448536269514722
_Z99 = 2.3263478740408408
_BURST_PROB = 0.06
_BURST_MEAN_SCALE = 2.5  # E[1 + Pareto(alpha=3)]
_BURST_SECOND_SCALE = 7.0  # E[(1 + Pareto(alpha=3))^2]
_TILT_SIGMA = 0.30
_ROLLOFF_MEAN = 0.5 * (0.0005 + 0.004)


def _bessel_i0(value: float) -> float:
    x = abs(value)
    if x < 3.75:
        t = x / 3.75
        t2 = t * t
        return 1.0 + t2 * (
            3.5156229
            + t2
            * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))
        )
    t = 3.75 / x
    return (math.exp(x) / math.sqrt(x)) * (
        0.39894228
        + t
        * (
            0.01328592
            + t
            * (
                0.00225319
                + t
                * (
                    -0.00157565
                    + t
                    * (
                        0.00916281
                        + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))
                    )
                )
            )
        )
    )


def _bessel_i1(value: float) -> float:
    x = abs(value)
    if x < 3.75:
        t = x / 3.75
        t2 = t * t
        ans = x * (
            0.5
            + t2
            * (
                0.87890594
                + t2
                * (
                    0.51498869
                    + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411)))
                )
            )
        )
    else:
        t = 3.75 / x
        ans = (math.exp(x) / math.sqrt(x)) * (
            0.39894228
            + t
            * (
                -0.03988024
                + t
                * (
                    -0.00362018
                    + t
                    * (
                        0.00163801
                        + t
                        * (
                            -0.01031555
                            + t
                            * (0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059)))
                        )
                    )
                )
            )
        )
    return ans if value >= 0 else -ans


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _amplitude_moments(config: AnalysisConfig, frequency_khz: float) -> tuple[float, float]:
    coherence = min(max(config.coherence, 0.0), 1.0)
    sigma_ln = 0.35 + 0.15 * (1.0 - coherence)
    log_ratio = math.log(max(frequency_khz, 1e-9) / 30.0)
    tilt_m1 = math.exp(0.5 * (_TILT_SIGMA * log_ratio) ** 2)
    tilt_m2 = math.exp(2.0 * (_TILT_SIGMA * log_ratio) ** 2)

    burst_m1 = (1.0 - _BURST_PROB) + _BURST_PROB * _BURST_MEAN_SCALE
    burst_m2 = (1.0 - _BURST_PROB) + _BURST_PROB * _BURST_SECOND_SCALE
    mean_amp = config.base_current_a * burst_m1 * tilt_m1
    second_moment_amp = (config.base_current_a**2) * math.exp(sigma_ln**2) * burst_m2 * tilt_m2
    return mean_amp, second_moment_amp


def _transfer_moments(
    config: AnalysisConfig, frequency_khz: float, n_points: int = 256
) -> tuple[float, float]:
    kernel = ExponentialKernel(alpha=config.kernel_alpha, resonance_scale=config.resonance_scale)
    radius = max(config.region_radius_m, 1e-9)
    admittance = config.admittance_s / (1.0 + _ROLLOFF_MEAN * max(frequency_khz - 30.0, 0.0))
    first = 0.0
    second = 0.0
    for idx in range(n_points):
        u = (idx + 0.5) / n_points
        distance = radius * math.sqrt(u)
        z_tr = kernel.impedance(frequency_khz, distance)
        gain = abs(z_tr) / abs(1.0 + admittance * z_tr)
        first += gain
        second += gain * gain
    return first / n_points, second / n_points


def _phase_pair_correlation(coherence: float) -> float:
    coherence_clamped = min(max(coherence, 0.0), 1.0)
    kappa = 0.2 + 26.0 * coherence_clamped
    denom = max(_bessel_i0(kappa), 1e-12)
    r1 = _bessel_i1(kappa) / denom
    return r1 * r1


def _lognormal_quantiles(mean_abs_v: float, var_v: float) -> tuple[float, float]:
    cv2 = var_v / max(mean_abs_v * mean_abs_v, 1e-12)
    sigma_sq = math.log1p(max(cv2, 0.0))
    sigma = math.sqrt(max(sigma_sq, 0.0))
    mu = math.log(max(mean_abs_v, 1e-12)) - 0.5 * sigma_sq
    p95 = math.exp(mu + _Z95 * sigma)
    p99 = math.exp(mu + _Z99 * sigma)
    return p95, p99


def _exceedance_probability_lognormal(mean_abs_v: float, var_v: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    cv2 = var_v / max(mean_abs_v * mean_abs_v, 1e-12)
    sigma_sq = math.log1p(max(cv2, 0.0))
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    mu = math.log(max(mean_abs_v, 1e-12)) - 0.5 * sigma_sq
    z = (math.log(threshold) - mu) / sigma
    return min(max(1.0 - _normal_cdf(z), 0.0), 1.0)


def compute_analytical_statistics(config: AnalysisConfig) -> StatisticsFrame:
    """Compute closed-form moment approximations over all configured frequencies."""
    config.validate()
    area_km2 = math.pi * (config.region_radius_m / 1000.0) ** 2
    mean_n = config.density * area_km2
    pair_correlation = _phase_pair_correlation(config.coherence)
    rows: StatisticsFrame = []
    density_scale = max(config.density, 1e-9) ** 0.5

    for frequency in config.frequencies_khz:
        mean_amp, second_amp = _amplitude_moments(config, frequency)
        transfer_m1, transfer_m2 = _transfer_moments(config, frequency)
        single_m1 = mean_amp * transfer_m1
        single_m2 = second_amp * transfer_m2

        second_moment_abs = mean_n * single_m2 + (mean_n**2) * pair_correlation * (single_m1**2)
        rms_abs_v = math.sqrt(max(second_moment_abs, 1e-12))

        # Approximate |V| mean from second moment with coherence-dependent correction.
        mean_scale = min(0.84 + 0.10 * pair_correlation, 0.98)
        mean_abs_v = mean_scale * rms_abs_v
        var_v = max(second_moment_abs - mean_abs_v * mean_abs_v, 1e-12)

        p95_abs_v, p99_abs_v = _lognormal_quantiles(mean_abs_v, var_v)
        tail_threshold = adaptive_threshold(
            floor_threshold=config.threshold,
            rms_abs_v=rms_abs_v,
            multiplier=config.threshold_rms_multiplier,
        )
        exceedance_probability = _exceedance_probability_lognormal(
            mean_abs_v=mean_abs_v,
            var_v=var_v,
            threshold=tail_threshold,
        )
        rows.append(
            {
                "frequency_khz": frequency,
                "mean_abs_v": mean_abs_v,
                "var_v": var_v,
                "rms_abs_v": rms_abs_v,
                "p95_abs_v": p95_abs_v,
                "p99_abs_v": p99_abs_v,
                "exceedance_probability": exceedance_probability,
                "exceedance_threshold_v": tail_threshold,
                "density_scaling": density_scale,
            }
        )
    return rows
