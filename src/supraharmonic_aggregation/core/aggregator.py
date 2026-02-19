"""Aggregation engine for complex PCC voltage."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Iterable

from .kernel import PropagationKernel


@dataclass(slots=True)
class Source:
    """A spatial source with electrical mark."""

    distance_m: float
    mark: object


SourcePopulation = list[Source]


class SupraharmonicAggregator:
    """Compute aggregate complex voltage from source populations."""

    def __init__(self, kernel: PropagationKernel) -> None:
        self.kernel = kernel

    def aggregate_complex_voltage(self, frequency_khz: float, sources: SourcePopulation) -> complex:
        """Aggregate complex source contributions at one frequency."""
        total = 0j
        for source in sources:
            amplitude = float(getattr(source.mark, "amplitude_a"))
            phase = float(getattr(source.mark, "phase_rad"))
            admittance = float(getattr(source.mark, "admittance_s"))
            current = amplitude * cmath.exp(1j * phase)
            z_tr = self.kernel.impedance(frequency_khz, source.distance_m)
            denominator = 1 + admittance * z_tr
            if abs(denominator) < 1e-12:
                continue
            total += (z_tr * current) / denominator
        return total

    def aggregate_magnitude(self, frequency_khz: float, sources: SourcePopulation) -> float:
        """Return absolute aggregate voltage magnitude."""
        return abs(self.aggregate_complex_voltage(frequency_khz, sources))
