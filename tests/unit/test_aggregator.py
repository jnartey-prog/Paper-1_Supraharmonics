from __future__ import annotations

import pytest

from supraharmonic_aggregation.core.aggregator import Source, SupraharmonicAggregator
from supraharmonic_aggregation.core.kernel import ExponentialKernel
from supraharmonic_aggregation.core.marks import SourceMark


@pytest.mark.unit
def test_aggregator_outputs_complex_voltage() -> None:
    kernel = ExponentialKernel(alpha=0.5, resonance_scale=0.0)
    aggregator = SupraharmonicAggregator(kernel)
    sources = [
        Source(distance_m=50.0, mark=SourceMark(amplitude_a=1.0, phase_rad=0.0, admittance_s=0.01)),
        Source(distance_m=120.0, mark=SourceMark(amplitude_a=0.8, phase_rad=1.0, admittance_s=0.01)),
    ]
    value = aggregator.aggregate_complex_voltage(10.0, sources)
    assert isinstance(value, complex)
    assert aggregator.aggregate_magnitude(10.0, sources) == abs(value)
