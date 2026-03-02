"""Source mark models and spatial population utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from .aggregator import Source


@dataclass(slots=True)
class SourceMark:
    """Electrical mark for one source."""

    amplitude_a: float
    phase_rad: float
    admittance_s: float
    spectral_tilt_per_decade: float = 0.0
    phase_slope_rad_per_khz: float = 0.0
    admittance_rolloff_per_khz: float = 0.0
    reference_frequency_khz: float = 30.0

    def amplitude_at_frequency(self, frequency_khz: float) -> float:
        """Return frequency-shaped current amplitude for this source."""
        f_ref = max(self.reference_frequency_khz, 1e-9)
        ratio = max(frequency_khz, 1e-9) / f_ref
        shaped = self.amplitude_a * (ratio**self.spectral_tilt_per_decade)
        return max(shaped, 1e-9)

    def phase_at_frequency(self, frequency_khz: float) -> float:
        """Return frequency-dependent phase with source-specific slope."""
        return self.phase_rad + self.phase_slope_rad_per_khz * (
            frequency_khz - self.reference_frequency_khz
        )

    def admittance_at_frequency(self, frequency_khz: float) -> float:
        """Return effective shunt admittance with mild high-frequency rolloff."""
        delta = max(frequency_khz - self.reference_frequency_khz, 0.0)
        scale = 1.0 + self.admittance_rolloff_per_khz * delta
        return max(self.admittance_s / scale, 0.0)


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Sample from Poisson distribution without hard clipping."""
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


def sample_mark(
    rng: random.Random,
    coherence: float,
    base_current_a: float,
    admittance_s: float,
    common_phase: float,
) -> SourceMark:
    """Sample one source mark with heavy-tailed amplitudes and spectral variability."""
    coherence_clamped = min(max(coherence, 0.0), 1.0)
    concentration = 0.2 + 26.0 * coherence_clamped
    phase = rng.vonmisesvariate(common_phase, concentration)

    sigma_ln = 0.35 + 0.15 * (1.0 - coherence_clamped)
    mu_ln = math.log(max(base_current_a, 1e-9)) - 0.5 * sigma_ln * sigma_ln
    amplitude = rng.lognormvariate(mu_ln, sigma_ln)
    if rng.random() < 0.06:
        amplitude *= 1.0 + rng.paretovariate(3.0)
    amplitude = min(max(amplitude, 1e-6), max(base_current_a, 1e-6) * 40.0)

    spectral_tilt = rng.gauss(0.0, 0.30)
    phase_slope = rng.gauss(0.0, 0.025)
    admittance_rolloff = rng.uniform(0.0005, 0.004)
    return SourceMark(
        amplitude_a=amplitude,
        phase_rad=phase,
        admittance_s=max(admittance_s, 0.0),
        spectral_tilt_per_decade=spectral_tilt,
        phase_slope_rad_per_khz=phase_slope,
        admittance_rolloff_per_khz=admittance_rolloff,
    )


def generate_source_population(
    density: float,
    region_radius_m: float,
    coherence: float,
    base_current_a: float,
    admittance_s: float,
    rng: random.Random,
) -> list[Source]:
    """Generate a PPP-like source set in a bounded circular region."""
    area_km2 = math.pi * (region_radius_m / 1000.0) ** 2
    n_sources = _sample_poisson(density * area_km2, rng)
    common_phase = rng.uniform(0.0, 2.0 * math.pi)
    population: list[Source] = []
    for _ in range(n_sources):
        distance = region_radius_m * math.sqrt(rng.random())
        mark = sample_mark(rng, coherence, base_current_a, admittance_s, common_phase)
        population.append(Source(distance_m=distance, mark=mark))
    return population


def amplitudes(population: Iterable[Source]) -> list[float]:
    """Extract source amplitudes."""
    values: list[float] = []
    for source in population:
        values.append(float(getattr(source.mark, "amplitude_a", 0.0)))
    return values
