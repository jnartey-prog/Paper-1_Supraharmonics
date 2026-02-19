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
