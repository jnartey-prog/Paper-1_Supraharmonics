"""Aggregation engine for complex PCC voltage."""

from __future__ import annotations

import cmath
from dataclasses import dataclass

from .kernel import PropagationKernel


@dataclass(slots=True)
class Source:
    """A spatial source with electrical mark."""

    distance_m: float
    mark: object


SourcePopulation = list[Source]


class SupraharmonicAggregator:
    """Compute aggregate complex voltage from source populations."""

    def __init__(self, kernel: PropagationKernel, min_denominator_magnitude: float = 1e-6) -> None:
        self.kernel = kernel
        self.min_denominator_magnitude = max(min_denominator_magnitude, 1e-12)

    @staticmethod
    def _resolve_mark_value(
        mark: object,
        frequency_khz: float,
        method_name: str,
        attr_name: str,
    ) -> float:
        method = getattr(mark, method_name, None)
        if callable(method):
            return float(method(frequency_khz))
        return float(getattr(mark, attr_name))

    def _regularize_denominator(self, denominator: complex) -> complex:
        magnitude = abs(denominator)
        if magnitude >= self.min_denominator_magnitude:
            return denominator
        if magnitude <= 0.0:
            return complex(self.min_denominator_magnitude, 0.0)
        return denominator * (self.min_denominator_magnitude / magnitude)

    def aggregate_complex_voltage(self, frequency_khz: float, sources: SourcePopulation) -> complex:
        """Aggregate complex source contributions at one frequency."""
        total = 0j
        for source in sources:
            amplitude = self._resolve_mark_value(
                source.mark,
                frequency_khz,
                method_name="amplitude_at_frequency",
                attr_name="amplitude_a",
            )
            phase = self._resolve_mark_value(
                source.mark,
                frequency_khz,
                method_name="phase_at_frequency",
                attr_name="phase_rad",
            )
            admittance = self._resolve_mark_value(
                source.mark,
                frequency_khz,
                method_name="admittance_at_frequency",
                attr_name="admittance_s",
            )
            current = amplitude * cmath.exp(1j * phase)
            z_tr = self.kernel.impedance(frequency_khz, source.distance_m)
            denominator = self._regularize_denominator(1 + admittance * z_tr)
            total += (z_tr * current) / denominator
        return total

    def aggregate_magnitude(self, frequency_khz: float, sources: SourcePopulation) -> float:
        """Return absolute aggregate voltage magnitude."""
        return abs(self.aggregate_complex_voltage(frequency_khz, sources))
