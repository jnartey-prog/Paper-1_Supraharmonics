from __future__ import annotations

from pathlib import Path

import pytest

from supraharmonic_aggregation.analysis.analytical import compute_analytical_statistics
from supraharmonic_aggregation.analysis.tail import compute_tail_metrics
from supraharmonic_aggregation.analysis.validation import check_integrability_conditions
from supraharmonic_aggregation.api import analyze, generate_artifacts
from supraharmonic_aggregation.simulation.monte_carlo import MonteCarloRunner


@pytest.mark.acceptance
def test_objective_001_model_formulation(baseline_config) -> None:
    rows = compute_analytical_statistics(baseline_config)
    assert len(rows) == len(baseline_config.frequencies_khz)


@pytest.mark.acceptance
def test_objective_002_first_second_order_stats(baseline_config) -> None:
    rows = compute_analytical_statistics(baseline_config)
    assert all("mean_abs_v" in row and "var_v" in row for row in rows)


@pytest.mark.acceptance
def test_objective_003_integrability_and_boundedness(baseline_config) -> None:
    report = check_integrability_conditions(baseline_config)
    assert report.finite_domain_ok and report.asymptotic_domain_ok


@pytest.mark.acceptance
def test_objective_004_tail_metrics() -> None:
    tail = compute_tail_metrics([0.1, 0.2, 0.5, 0.9, 1.4], threshold=0.7)
    assert tail.percentiles[95] >= tail.percentiles[90]
    assert tail.exceedance_probability is not None


@pytest.mark.acceptance
def test_objective_005_mc_and_benchmark_validation(baseline_config) -> None:
    mc = MonteCarloRunner(baseline_config).run(20)
    assert len(mc.statistics_frame) == len(baseline_config.frequencies_khz)


@pytest.mark.acceptance
def test_objective_006_reproducible_artifacts_and_metadata(baseline_config) -> None:
    run = analyze(baseline_config)
    paths = generate_artifacts(run, output_dir=baseline_config.output_dir)
    assert len(paths) == 14
    assert all(Path(path).exists() for path in paths)
