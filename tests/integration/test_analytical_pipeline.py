from __future__ import annotations

import pytest

from supraharmonic_aggregation.api import analyze


@pytest.mark.integration
def test_analyze_returns_run_bundle(baseline_config) -> None:
    run = analyze(baseline_config)
    assert run.run_id
    assert run.analytical
    assert run.monte_carlo.statistics_frame
