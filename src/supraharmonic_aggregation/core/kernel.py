"""Propagation kernel abstractions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class PropagationKernel(Protocol):
    """Kernel protocol for transfer impedance models."""

    def impedance(self, frequency_khz: float, distance_m: float) -> complex:
        """Return complex transfer impedance at frequency and distance."""


@dataclass(slots=True)
class ExponentialKernel:
    """Simple attenuation kernel with optional weak resonance modulation."""

    alpha: float
    resonance_scale: float = 0.0

    def impedance(self, frequency_khz: float, distance_m: float) -> complex:
        """Compute bounded complex impedance."""
        distance_km = max(distance_m, 1e-9) / 1000.0
        attenuation = math.exp(-self.alpha * distance_km)
        resonance = 1.0 + self.resonance_scale * math.sin(frequency_khz / 12.5)
        phase = frequency_khz * 0.0025 * distance_km
        magnitude = max(attenuation * resonance, 1e-9)
        return magnitude * complex(math.cos(phase), math.sin(phase))
