"""Monte Carlo runner for aggregation experiments."""

from __future__ import annotations

import random

from ..analysis.tail import compute_tail_metrics
from ..config import AnalysisConfig
from ..core.aggregator import SupraharmonicAggregator
from ..core.kernel import ExponentialKernel
from ..core.marks import generate_source_population
from ..models import MonteCarloResult


class MonteCarloRunner:
    """Run stochastic simulation with deterministic seed controls."""

    def __init__(self, config: AnalysisConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed

    def run(self, n_samples: int) -> MonteCarloResult:
        """Run Monte Carlo simulations and summarize per-frequency outputs."""
        self.config.validate()
        rng = random.Random(self.seed)
        kernel = ExponentialKernel(alpha=self.config.kernel_alpha, resonance_scale=self.config.resonance_scale)
        aggregator = SupraharmonicAggregator(kernel)
        per_frequency_samples: dict[str, list[float]] = {
            str(freq): [] for freq in self.config.frequencies_khz
        }

        for _ in range(n_samples):
            population = generate_source_population(
                density=self.config.density,
                region_radius_m=self.config.region_radius_m,
                coherence=self.config.coherence,
                base_current_a=self.config.base_current_a,
                admittance_s=self.config.admittance_s,
                rng=rng,
            )
            for frequency in self.config.frequencies_khz:
                value = aggregator.aggregate_magnitude(frequency, population)
                per_frequency_samples[str(frequency)].append(value)

        statistics_frame: list[dict[str, float | int | str]] = []
        for frequency in self.config.frequencies_khz:
            key = str(frequency)
            values = per_frequency_samples[key]
            tail = compute_tail_metrics(values, threshold=self.config.threshold)
            mean_abs_v = sum(values) / len(values) if values else 0.0
            var_v = (
                sum((value - mean_abs_v) ** 2 for value in values) / len(values)
                if values
                else 0.0
            )
            statistics_frame.append(
                {
                    "frequency_khz": frequency,
                    "mean_abs_v": mean_abs_v,
                    "var_v": var_v,
                    "rms_abs_v": (mean_abs_v**2 + var_v) ** 0.5,
                    "p90_abs_v": tail.percentiles.get(90, 0.0),
                    "p95_abs_v": tail.percentiles.get(95, 0.0),
                    "p99_abs_v": tail.percentiles.get(99, 0.0),
                    "exceedance_probability": tail.exceedance_probability or 0.0,
                    "sample_size": tail.sample_size,
                }
            )

        return MonteCarloResult(
            per_frequency_samples=per_frequency_samples,
            statistics_frame=statistics_frame,
        )
