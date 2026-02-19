"""Tail and exceedance metric computations."""

from __future__ import annotations

from typing import Iterable

from ..models import TailMetrics


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    weight = position - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def compute_tail_metrics(
    samples: Iterable[float],
    percentiles: tuple[int, ...] = (90, 95, 99),
    threshold: float | None = None,
) -> TailMetrics:
    """Compute percentile and exceedance metrics from sample values."""
    values = sorted(float(value) for value in samples)
    percentile_map: dict[int, float] = {}
    for percentile in percentiles:
        percentile_map[percentile] = _quantile(values, percentile / 100.0)
    exceedance_probability: float | None = None
    if threshold is not None and values:
        exceedance_probability = sum(1 for value in values if value > threshold) / len(values)
    return TailMetrics(
        percentiles=percentile_map,
        exceedance_probability=exceedance_probability,
        sample_size=len(values),
    )
