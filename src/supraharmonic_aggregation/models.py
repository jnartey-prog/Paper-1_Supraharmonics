"""Shared typed data models used across the package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


StatisticsFrame = list[dict[str, float | int | str]]


@dataclass(slots=True)
class TailMetrics:
    """Tail metrics for magnitude distributions."""

    percentiles: dict[int, float]
    exceedance_probability: float | None
    sample_size: int


@dataclass(slots=True)
class IntegrabilityReport:
    """Result of boundedness/integrability checks."""

    finite_domain_ok: bool
    asymptotic_domain_ok: bool
    details: str


@dataclass(slots=True)
class MonteCarloResult:
    """Monte Carlo simulation outputs."""

    per_frequency_samples: dict[str, list[float]]
    statistics_frame: StatisticsFrame


@dataclass(slots=True)
class BenchmarkComparison:
    """Analytical vs benchmark comparison results."""

    rows: StatisticsFrame


@dataclass(slots=True)
class RunBundle:
    """Container returned by end-to-end analysis workflows."""

    run_id: str
    config: dict[str, Any]
    analytical: StatisticsFrame
    monte_carlo: MonteCarloResult
    benchmark: BenchmarkComparison
    artifact_paths: list[str] = field(default_factory=list)
    run_manifest_path: str | None = None
    log_path: str | None = None
    stats_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this run bundle to a JSON-compatible dict."""
        payload = asdict(self)
        payload["monte_carlo"] = {
            "per_frequency_samples": self.monte_carlo.per_frequency_samples,
            "statistics_frame": self.monte_carlo.statistics_frame,
        }
        payload["benchmark"] = {"rows": self.benchmark.rows}
        return payload
