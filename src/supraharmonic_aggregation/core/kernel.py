"""Propagation kernel abstractions."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Protocol
from typing import Literal


class PropagationKernel(Protocol):
    """Kernel protocol for transfer impedance models."""

    def impedance(self, frequency_khz: float, distance_m: float) -> complex:
        """Return complex transfer impedance at frequency and distance."""


@dataclass(slots=True)
class ExponentialKernel:
    """Finite-line RLGC transfer kernel with bounded resonance emphasis."""

    alpha: float
    resonance_scale: float = 0.0
    r_ohm_per_km: float = 0.35
    l_h_per_km: float = 0.45e-3
    c_f_per_km: float = 120e-9
    g_s_per_km: float = 0.0
    source_impedance_ohm: float = 0.03
    load_impedance_ohm: float = 0.30
    feeder_length_km: float = 1.0
    skin_effect_coeff: float = 0.18
    skin_effect_ref_hz: float = 2000.0
    dielectric_tan_delta: float = 0.015
    termination_mode: Literal["matched", "resistive"] = "matched"

    @staticmethod
    def _line_input_impedance(
        gamma: complex, zc: complex, length_km: float, z_term: complex
    ) -> complex:
        if length_km <= 1e-12:
            return z_term
        t = cmath.tanh(gamma * length_km)
        denom = zc + z_term * t
        if abs(denom) <= 1e-18:
            return zc
        return zc * (z_term + zc * t) / denom

    def _resonance_gain(self, frequency_hz: float) -> float:
        if self.resonance_scale <= 0:
            return 1.0
        lc = max(self.l_h_per_km * self.c_f_per_km, 1e-18)
        f0_hz = 1.0 / (2.0 * math.pi * math.sqrt(lc))
        width_hz = max(0.2 * f0_hz, 1.0)
        detuning = (frequency_hz - f0_hz) / width_hz
        return 1.0 + self.resonance_scale / (1.0 + detuning * detuning)

    def impedance(self, frequency_khz: float, distance_m: float) -> complex:
        """Compute bounded complex transfer impedance for a source at distance d from PCC."""
        distance_km = max(distance_m, 0.0) / 1000.0
        frequency_hz = max(frequency_khz, 1e-9) * 1000.0
        omega = 2.0 * math.pi * frequency_hz

        skin_ref_hz = max(self.skin_effect_ref_hz, 1.0)
        skin_mult = 1.0 + self.skin_effect_coeff * math.sqrt(frequency_hz / skin_ref_hz)
        r_ac = max(self.r_ohm_per_km, 1e-12) * skin_mult
        g_dielectric = omega * max(self.c_f_per_km, 0.0) * max(self.dielectric_tan_delta, 0.0)

        series_impedance_per_km = complex(r_ac, omega * self.l_h_per_km)
        shunt_admittance_per_km = complex(self.g_s_per_km + g_dielectric, omega * self.c_f_per_km)
        propagation = cmath.sqrt(series_impedance_per_km * shunt_admittance_per_km) + complex(
            self.alpha, 0.0
        )
        characteristic = cmath.sqrt(series_impedance_per_km / shunt_admittance_per_km)

        total_len_km = max(self.feeder_length_km, distance_km)
        left_len_km = min(distance_km, total_len_km)
        right_len_km = max(total_len_km - left_len_km, 0.0)

        z_source = complex(max(self.source_impedance_ohm, 1e-9), 0.0)
        if self.termination_mode == "matched":
            # Matched termination suppresses standing-wave artifacts in benchmark views.
            z_right_term = characteristic
        else:
            z_right_term = complex(max(self.load_impedance_ohm, 1e-9), 0.0)

        zin_left = self._line_input_impedance(propagation, characteristic, left_len_km, z_source)
        zin_right = self._line_input_impedance(
            propagation, characteristic, right_len_km, z_right_term
        )
        zin_sum = zin_left + zin_right
        if abs(zin_sum) <= 1e-18:
            z_parallel = zin_left
        else:
            z_parallel = (zin_left * zin_right) / zin_sum

        # ABCD relation for the left segment from source node to PCC:
        # Vs = (A + B/Zs) * Vpcc  ->  Vpcc = Vs / (A + B/Zs)
        a = cmath.cosh(propagation * left_len_km)
        b = characteristic * cmath.sinh(propagation * left_len_km)
        transfer_den = a + (b / z_source)
        if abs(transfer_den) <= 1e-18:
            transfer_den = complex(1e-18, 0.0)

        value = (z_parallel / transfer_den) * self._resonance_gain(frequency_hz)
        magnitude = max(abs(value), 1e-9)
        if abs(value) <= 1e-12:
            return complex(magnitude, 0.0)
        return value * (magnitude / abs(value))
