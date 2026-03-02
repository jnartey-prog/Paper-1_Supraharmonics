"""Robustness studies for reviewer-oriented validation stress tests."""

from __future__ import annotations

import statistics

from ..benchmark.compare import compare_with_feeder_benchmark
from ..benchmark.independent import IndependentBenchmarkRunner
from ..config import AnalysisConfig
from ..simulation.synthetic_data import SyntheticDataGenerator
from .analytical import compute_analytical_statistics


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


def summarize_validation_errors(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    """Summarize benchmark relative errors into reviewer-facing scalar diagnostics."""
    rms = [float(row["relative_error_rms"]) for row in rows]
    p95 = [float(row["relative_error_p95"]) for row in rows]
    return {
        "rms_error_mean": statistics.fmean(rms) if rms else 0.0,
        "rms_error_median": statistics.median(rms) if rms else 0.0,
        "rms_error_p90": _quantile(rms, 0.90),
        "rms_error_max": max(rms) if rms else 0.0,
        "p95_error_mean": statistics.fmean(p95) if p95 else 0.0,
        "p95_error_median": statistics.median(p95) if p95 else 0.0,
        "p95_error_p90": _quantile(p95, 0.90),
        "p95_error_max": max(p95) if p95 else 0.0,
    }


def run_multiseed_validation_study(
    config: AnalysisConfig,
    seeds: list[int],
    n_samples: int,
    frequencies_khz: list[float] | None = None,
) -> list[dict[str, float | int | str]]:
    """Run multi-seed internal and independent benchmark validation summaries."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not seeds:
        raise ValueError("seeds must not be empty.")

    cfg = config
    if frequencies_khz is not None:
        cfg = AnalysisConfig.from_dict(
            {**config.to_dict(), "frequencies_khz": list(frequencies_khz)}
        )

    analytical = compute_analytical_statistics(cfg)
    out: list[dict[str, float | int | str]] = []
    for seed in seeds:
        synthetic = SyntheticDataGenerator(cfg, seed=seed).generate(
            n_samples=n_samples,
            include_complex=False,
            frequencies_khz=cfg.frequencies_khz,
            include_measurement_noise=True,
        )
        internal_summary = summarize_validation_errors(synthetic.validation_frame)

        independent = IndependentBenchmarkRunner(cfg, seed=seed + 1_000_003).run(
            n_samples=n_samples,
            frequencies_khz=cfg.frequencies_khz,
        )
        independent_validation = compare_with_feeder_benchmark(
            analytical=analytical,
            simulated=independent.statistics_frame,
        ).rows
        independent_summary = summarize_validation_errors(independent_validation)

        row: dict[str, float | int | str] = {
            "seed": seed,
            "n_frequencies": len(cfg.frequencies_khz),
            "n_samples_per_frequency": n_samples,
            "internal_rms_error_mean": internal_summary["rms_error_mean"],
            "internal_rms_error_p90": internal_summary["rms_error_p90"],
            "internal_p95_error_mean": internal_summary["p95_error_mean"],
            "internal_p95_error_p90": internal_summary["p95_error_p90"],
            "independent_rms_error_mean": independent_summary["rms_error_mean"],
            "independent_rms_error_p90": independent_summary["rms_error_p90"],
            "independent_p95_error_mean": independent_summary["p95_error_mean"],
            "independent_p95_error_p90": independent_summary["p95_error_p90"],
            "internal_pass_strict": (
                internal_summary["rms_error_p90"] <= 0.10
                and internal_summary["p95_error_p90"] <= 0.10
            ),
            "independent_pass_strict": (
                independent_summary["rms_error_p90"] <= 0.20
                and independent_summary["p95_error_p90"] <= 0.20
            ),
        }
        out.append(row)
    return out


def summarize_multiseed_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    """Aggregate multi-seed rows into one compact summary row."""
    if not rows:
        return {}
    internal_rms = [float(row["internal_rms_error_mean"]) for row in rows]
    internal_p95 = [float(row["internal_p95_error_mean"]) for row in rows]
    independent_rms = [float(row["independent_rms_error_mean"]) for row in rows]
    independent_p95 = [float(row["independent_p95_error_mean"]) for row in rows]
    return {
        "n_seeds": len(rows),
        "internal_rms_error_mean_over_seeds": statistics.fmean(internal_rms),
        "internal_rms_error_max_over_seeds": max(internal_rms),
        "internal_p95_error_mean_over_seeds": statistics.fmean(internal_p95),
        "internal_p95_error_max_over_seeds": max(internal_p95),
        "independent_rms_error_mean_over_seeds": statistics.fmean(independent_rms),
        "independent_rms_error_max_over_seeds": max(independent_rms),
        "independent_p95_error_mean_over_seeds": statistics.fmean(independent_p95),
        "independent_p95_error_max_over_seeds": max(independent_p95),
        "internal_pass_count": sum(1 for row in rows if bool(row["internal_pass_strict"])),
        "independent_pass_count": sum(1 for row in rows if bool(row["independent_pass_strict"])),
    }
