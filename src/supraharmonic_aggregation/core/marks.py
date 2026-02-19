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


def _sample_poisson(lam: float, rng: random.Random) -> int:
    """Sample from Poisson distribution using Knuth algorithm."""
    if lam <= 0:
        return 0
    lam = min(lam, 500.0)
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return max(k - 1, 0)


def _sample_mark(
    rng: random.Random,
    coherence: float,
    base_current_a: float,
    admittance_s: float,
    common_phase: float,
) -> SourceMark:
    """Sample one source mark with coherence-controlled phase."""
    residual = rng.uniform(0.0, 2.0 * math.pi)
    phase = coherence * common_phase + (1.0 - coherence) * residual
    amplitude = max(base_current_a * (0.7 + 0.6 * rng.random()), 1e-6)
    return SourceMark(amplitude_a=amplitude, phase_rad=phase, admittance_s=admittance_s)


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
        mark = _sample_mark(rng, coherence, base_current_a, admittance_s, common_phase)
        population.append(Source(distance_m=distance, mark=mark))
    return population


def amplitudes(population: Iterable[Source]) -> list[float]:
    """Extract source amplitudes."""
    return [source.mark.amplitude_a for source in population]
