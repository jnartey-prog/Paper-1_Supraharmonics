from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.analytical import compute_analytical_statistics
from supraharmonic_aggregation.simulation.monte_carlo import MonteCarloRunner


@pytest.mark.integration
def test_mc_and_analytical_share_frequency_grid(baseline_config) -> None:
    analytical = compute_analytical_statistics(baseline_config)
    mc = MonteCarloRunner(baseline_config, seed=baseline_config.seed).run(24)
    analytical_freq = {row["frequency_khz"] for row in analytical}
    mc_freq = {row["frequency_khz"] for row in mc.statistics_frame}
    assert analytical_freq == mc_freq
