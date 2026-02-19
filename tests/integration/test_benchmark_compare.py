from __future__ import annotations

import pytest

from supraharmonic_aggregation.benchmark.compare import compare_with_feeder_benchmark


@pytest.mark.integration
def test_benchmark_compare_produces_relative_errors() -> None:
    analytical = [
        {"frequency_khz": 10.0, "rms_abs_v": 1.2, "p95_abs_v": 1.8},
        {"frequency_khz": 30.0, "rms_abs_v": 1.5, "p95_abs_v": 2.1},
    ]
    simulated = [
        {"frequency_khz": 10.0, "rms_abs_v": 1.0, "p95_abs_v": 1.6},
        {"frequency_khz": 30.0, "rms_abs_v": 1.7, "p95_abs_v": 2.2},
    ]
    result = compare_with_feeder_benchmark(analytical, simulated)
    assert len(result.rows) == 2
    assert all("relative_error_rms" in row for row in result.rows)
