from __future__ import annotations

import pytest

from supraharmonic_aggregation.api import analyze, generate_artifacts


@pytest.mark.integration
def test_artifact_generator_outputs_required_counts(baseline_config) -> None:
    run = analyze(baseline_config)
    paths = generate_artifacts(run, output_dir=baseline_config.output_dir)
    table_count = len([path for path in paths if "table_" in path])
    figure_count = len([path for path in paths if "figure_" in path])
    assert table_count == 6
    assert figure_count == 8
