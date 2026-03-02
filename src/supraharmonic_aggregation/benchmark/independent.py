"""Independent benchmark generator with alternative modeling assumptions."""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass

from ..analysis.tail import adaptive_threshold, compute_tail_metrics
from ..config import AnalysisConfig
from ..core.kernel import ExponentialKernel
from ..models import StatisticsFrame


@dataclass(slots=True)
class IndependentBenchmarkResult:
    """Independent benchmark outputs for cross-model validation."""

    per_frequency_samples: dict[str, list[float]]
    statistics_frame: StatisticsFrame


def _sample_poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam <= 40.0:
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while p > limit:
            k += 1
            p *= rng.random()
        return max(k - 1, 0)
    if lam <= 5_000.0:
        whole = int(lam // 40.0)
        remainder = lam - (whole * 40.0)
        total = 0
        for _ in range(whole):
            total += _sample_poisson(40.0, rng)
        if remainder > 0:
            total += _sample_poisson(remainder, rng)
        return total
    draw = int(round(rng.gauss(lam, math.sqrt(lam))))
    return max(draw, 0)


def _sample_count(mean_count: float, overdispersion_k: float, rng: random.Random) -> int:
    scale = mean_count / max(overdispersion_k, 1e-9)
    gamma_rate = rng.gammavariate(overdispersion_k, scale)
    return _sample_poisson(gamma_rate, rng)


def _sample_distance(region_radius_m: float, rng: random.Random) -> float:
    # Mixture of diffuse and hotspot radial patterns to break PPP-only assumptions.
    if rng.random() < 0.82:
        return region_radius_m * math.sqrt(rng.random())
    hotspot = rng.betavariate(2.5, 6.0)
    return region_radius_m * hotspot


def _independent_kernel(config: AnalysisConfig) -> ExponentialKernel:
    return ExponentialKernel(
        alpha=config.kernel_alpha * 1.12 + 0.03,
        resonance_scale=max(config.resonance_scale * 0.55, 0.01),
        r_ohm_per_km=0.41,
        l_h_per_km=0.50e-3,
        c_f_per_km=105e-9,
        source_impedance_ohm=0.05,
        load_impedance_ohm=0.33,
    )


def _frequency_correction(frequency_khz: float) -> float:
    low_edge = 0.83 * math.exp(-frequency_khz / 3.8)
    high_sigmoid = 1.0 / (1.0 + math.exp(-(frequency_khz - 70.0) / 12.0))
    high_edge = 0.275 * ((max(frequency_khz, 1e-9) / 150.0) ** 0.70) * high_sigmoid
    return 1.0 + low_edge + high_edge


def _summarize_statistics(
    frequencies_khz: list[float],
    per_frequency_samples: dict[str, list[float]],
    threshold: float,
    threshold_rms_multiplier: float,
) -> StatisticsFrame:
    rows: StatisticsFrame = []
    for frequency in frequencies_khz:
        key = str(frequency)
        values = per_frequency_samples[key]
        mean_abs_v = sum(values) / len(values) if values else 0.0
        var_v = sum((value - mean_abs_v) ** 2 for value in values) / len(values) if values else 0.0
        rms_abs_v = math.sqrt(max(mean_abs_v * mean_abs_v + var_v, 0.0))
        tail_threshold = adaptive_threshold(
            floor_threshold=threshold,
            rms_abs_v=rms_abs_v,
            multiplier=threshold_rms_multiplier,
        )
        tail = compute_tail_metrics(values, threshold=tail_threshold)
        rows.append(
            {
                "frequency_khz": frequency,
                "mean_abs_v": mean_abs_v,
                "var_v": var_v,
                "rms_abs_v": rms_abs_v,
                "p90_abs_v": tail.percentiles.get(90, 0.0),
                "p95_abs_v": tail.percentiles.get(95, 0.0),
                "p99_abs_v": tail.percentiles.get(99, 0.0),
                "exceedance_probability": tail.exceedance_probability or 0.0,
                "exceedance_threshold_v": tail_threshold,
                "sample_size": tail.sample_size,
            }
        )
    return rows


class IndependentBenchmarkRunner:
    """Generate a benchmark using an intentionally different model family."""

    def __init__(self, config: AnalysisConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    def run(
        self,
        n_samples: int,
        frequencies_khz: list[float] | None = None,
    ) -> IndependentBenchmarkResult:
        self.config.validate()
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        freqs = list(frequencies_khz or self.config.frequencies_khz)
        rng = random.Random(self.seed)
        area_km2 = math.pi * (self.config.region_radius_m / 1000.0) ** 2
        mean_count = self.config.density * area_km2
        overdispersion_k = 4.8
        kernel = _independent_kernel(self.config)

        per_frequency_samples: dict[str, list[float]] = {str(freq): [] for freq in freqs}
        for _ in range(n_samples):
            n_sources = _sample_count(mean_count, overdispersion_k, rng)
            common_phase = rng.uniform(0.0, 2.0 * math.pi)
            source_distances = [
                _sample_distance(self.config.region_radius_m, rng) for _ in range(n_sources)
            ]
            sigma_ln = 0.30 + 0.12 * (1.0 - self.config.coherence)
            mu_ln = math.log(max(self.config.base_current_a, 1e-9)) - 0.5 * sigma_ln * sigma_ln
            source_base_amplitudes = [
                min(
                    max(rng.lognormvariate(mu_ln, sigma_ln), 1e-6),
                    max(self.config.base_current_a, 1e-6) * 25.0,
                )
                for _ in range(n_sources)
            ]
            for idx in range(n_sources):
                if rng.random() < 0.03:
                    source_base_amplitudes[idx] *= 1.0 + rng.paretovariate(3.5)
            kappa = 0.35 + 18.0 * self.config.coherence
            source_phase_offsets = [
                rng.vonmisesvariate(common_phase, kappa) for _ in range(n_sources)
            ]
            source_admittances = [
                max(self.config.admittance_s * rng.lognormvariate(-0.03, 0.28), 1e-6)
                for _ in range(n_sources)
            ]

            for frequency in freqs:
                total = 0j
                log_ratio = math.log(max(frequency, 1e-9) / 30.0)
                for idx in range(n_sources):
                    tilt = math.exp(rng.gauss(0.0, 0.18) * log_ratio)
                    amplitude = source_base_amplitudes[idx] * tilt
                    phase = source_phase_offsets[idx] + rng.gauss(0.0, 0.018) * frequency
                    current = amplitude * cmath.exp(1j * phase)
                    z_tr = kernel.impedance(frequency, source_distances[idx])
                    damping = 1.0 + source_admittances[idx] * abs(z_tr) + 0.15 * abs(z_tr)
                    total += (z_tr * current) / max(damping, 1e-9)

                observed_abs_v = abs(total)
                observed_abs_v *= _frequency_correction(frequency)
                if self.config.measurement_noise_cv > 0:
                    observed_abs_v = observed_abs_v * (1.0 + self.config.measurement_bias)
                    observed_abs_v += rng.gauss(
                        0.0, self.config.measurement_noise_cv * max(observed_abs_v, 1e-6)
                    )
                    observed_abs_v = max(observed_abs_v, 0.0)
                per_frequency_samples[str(frequency)].append(observed_abs_v)

        statistics_frame = _summarize_statistics(
            frequencies_khz=freqs,
            per_frequency_samples=per_frequency_samples,
            threshold=self.config.threshold,
            threshold_rms_multiplier=self.config.threshold_rms_multiplier,
        )
        return IndependentBenchmarkResult(
            per_frequency_samples=per_frequency_samples,
            statistics_frame=statistics_frame,
        )
