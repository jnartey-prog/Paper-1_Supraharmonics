"""Closed/semi-closed analytical statistics approximations."""

from __future__ import annotations

import math

from ..config import AnalysisConfig
from ..core.kernel import ExponentialKernel
from ..models import StatisticsFrame


def _kernel_mean_abs(config: AnalysisConfig, frequency_khz: float) -> float:
    kernel = ExponentialKernel(alpha=config.kernel_alpha, resonance_scale=config.resonance_scale)
    radius = max(config.region_radius_m, 1.0)
    samples = []
    for idx in range(1, 65):
        distance = (idx / 64.0) * radius
        samples.append(abs(kernel.impedance(frequency_khz, distance)))
    return sum(samples) / len(samples)


def compute_analytical_statistics(config: AnalysisConfig) -> StatisticsFrame:
    """Compute deterministic analytical statistics for each configured frequency."""
    config.validate()
    rows: StatisticsFrame = []
    density_scale = max(config.density, 1e-9) ** 0.5
    for frequency in config.frequencies_khz:
        k_mean = _kernel_mean_abs(config, frequency)
        mean_abs_v = config.base_current_a * k_mean * config.coherence * config.density * 0.1
        var_v = (config.base_current_a**2) * (k_mean**2) * (1.0 - config.coherence**2) * config.density
        rms_abs_v = math.sqrt(max(var_v + mean_abs_v**2, 0.0))
        rows.append(
            {
                "frequency_khz": frequency,
                "mean_abs_v": mean_abs_v,
                "var_v": var_v,
                "rms_abs_v": rms_abs_v,
                "density_scaling": density_scale,
            }
        )
    return rows
