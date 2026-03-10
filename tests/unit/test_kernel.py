from __future__ import annotations

import math

import pytest

from supraharmonic_aggregation.core.kernel import ExponentialKernel


@pytest.mark.unit
def test_kernel_returns_finite_complex_value() -> None:
    kernel = ExponentialKernel(alpha=0.8, resonance_scale=0.1)
    value = kernel.impedance(30.0, 120.0)
    assert isinstance(value, complex)
    assert math.isfinite(value.real)
    assert math.isfinite(value.imag)
    assert abs(value) > 0


@pytest.mark.unit
def test_kernel_resonance_center_shift_changes_magnitude_profile() -> None:
    baseline = ExponentialKernel(alpha=0.8, resonance_scale=0.1)
    shifted = ExponentialKernel(alpha=0.8, resonance_scale=0.1, resonance_center_hz=45_000.0)
    baseline_mag = abs(baseline.impedance(45.0, 120.0))
    shifted_mag = abs(shifted.impedance(45.0, 120.0))
    assert shifted_mag > baseline_mag
