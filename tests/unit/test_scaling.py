from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.scaling import evaluate_scaling_laws


@pytest.mark.unit
def test_scaling_outputs_density_rows(baseline_config) -> None:
    densities = [5.0, 10.0, 20.0]
    rows = evaluate_scaling_laws(baseline_config, densities=densities, coherence=0.0)
    assert len(rows) == len(densities)
    assert rows[0]["density"] == 5.0
