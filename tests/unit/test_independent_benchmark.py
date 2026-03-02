from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.robustness import (
    run_multiseed_validation_study,
    summarize_multiseed_rows,
)
from supraharmonic_aggregation.benchmark.independent import IndependentBenchmarkRunner


@pytest.mark.unit
def test_independent_benchmark_runner_shape(baseline_config) -> None:
    result = IndependentBenchmarkRunner(baseline_config, seed=401).run(n_samples=16)
    assert len(result.statistics_frame) == len(baseline_config.frequencies_khz)
    assert all(int(row["sample_size"]) == 16 for row in result.statistics_frame)
    assert all("p95_abs_v" in row for row in result.statistics_frame)


@pytest.mark.unit
def test_multiseed_validation_study_outputs_internal_and_independent_rows(baseline_config) -> None:
    rows = run_multiseed_validation_study(
        config=baseline_config,
        seeds=[101, 103],
        n_samples=12,
        frequencies_khz=baseline_config.frequencies_khz,
    )
    summary = summarize_multiseed_rows(rows)
    assert len(rows) == 2
    assert all("internal_rms_error_mean" in row for row in rows)
    assert all("independent_rms_error_mean" in row for row in rows)
    assert int(summary["n_seeds"]) == 2
