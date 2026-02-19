from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.analytical import compute_analytical_statistics


@pytest.mark.unit
def test_analytical_statistics_shape(baseline_config) -> None:
    rows = compute_analytical_statistics(baseline_config)
    assert len(rows) == len(baseline_config.frequencies_khz)
    assert all("rms_abs_v" in row for row in rows)
