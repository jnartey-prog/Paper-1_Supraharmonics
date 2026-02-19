from __future__ import annotations

import pytest

from supraharmonic_aggregation.analysis.validation import check_integrability_conditions


@pytest.mark.unit
def test_integrability_report_passes_for_valid_config(baseline_config) -> None:
    report = check_integrability_conditions(baseline_config)
    assert report.finite_domain_ok
    assert report.asymptotic_domain_ok
