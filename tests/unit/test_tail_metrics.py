from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.tail import compute_tail_metrics


@pytest.mark.unit
def test_tail_metrics_percentiles_and_exceedance() -> None:
    samples = [0.2, 0.4, 0.6, 0.8, 1.0]
    result = compute_tail_metrics(samples, percentiles=(90, 95), threshold=0.5)
    assert result.sample_size == 5
    assert 90 in result.percentiles
    assert 95 in result.percentiles
    assert result.exceedance_probability == 3 / 5
