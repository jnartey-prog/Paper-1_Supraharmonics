"""Comparison utilities for analytical vs benchmark/simulated outputs."""

from __future__ import annotations

from ..models import BenchmarkComparison, StatisticsFrame


def _index_by_frequency(rows: StatisticsFrame) -> dict[float, dict[str, float | int | str]]:
    return {float(row["frequency_khz"]): row for row in rows}


def compare_with_feeder_benchmark(
    analytical: StatisticsFrame,
    simulated: StatisticsFrame,
) -> BenchmarkComparison:
    """Compute relative error metrics for overlapping frequency points."""
    analytical_idx = _index_by_frequency(analytical)
    simulated_idx = _index_by_frequency(simulated)
    rows: StatisticsFrame = []
    for frequency in sorted(set(analytical_idx).intersection(simulated_idx)):
        a_row = analytical_idx[frequency]
        s_row = simulated_idx[frequency]
        a_rms = float(a_row.get("rms_abs_v", 0.0))
        s_rms = float(s_row.get("rms_abs_v", 0.0))
        denom = max(abs(s_rms), 1e-9)
        rel_error_rms = abs(a_rms - s_rms) / denom
        a_p95 = float(a_row.get("p95_abs_v", a_rms))
        s_p95 = float(s_row.get("p95_abs_v", s_rms))
        rel_error_p95 = abs(a_p95 - s_p95) / max(abs(s_p95), 1e-9)
        rows.append(
            {
                "frequency_khz": frequency,
                "relative_error_rms": rel_error_rms,
                "relative_error_p95": rel_error_p95,
            }
        )
    return BenchmarkComparison(rows=rows)
