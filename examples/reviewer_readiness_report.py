"""Generate reviewer-readiness robustness artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import supraharmonic_aggregation as sha
from supraharmonic_aggregation.analysis.analytical import compute_analytical_statistics
from supraharmonic_aggregation.analysis.robustness import (
    run_multiseed_validation_study,
    summarize_multiseed_rows,
    summarize_validation_errors,
)
from supraharmonic_aggregation.benchmark.compare import compare_with_feeder_benchmark
from supraharmonic_aggregation.benchmark.independent import IndependentBenchmarkRunner
from supraharmonic_aggregation.config import AnalysisConfig


def _write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _full_integer_grid() -> list[float]:
    return [float(value) for value in range(2, 151)]


def _report_markdown(
    path: Path,
    multiseed_summary: dict[str, float | int],
    independent_summary: dict[str, float],
    multiseed_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Reviewer-Readiness Validation Summary",
        "",
        f"- Seeds evaluated: {int(multiseed_summary['n_seeds'])}",
        f"- Internal strict passes: {int(multiseed_summary['internal_pass_count'])}/{int(multiseed_summary['n_seeds'])}",
        f"- Independent strict passes: {int(multiseed_summary['independent_pass_count'])}/{int(multiseed_summary['n_seeds'])}",
        f"- Internal RMS mean over seeds: {float(multiseed_summary['internal_rms_error_mean_over_seeds']):.4f}",
        f"- Internal P95 mean over seeds: {float(multiseed_summary['internal_p95_error_mean_over_seeds']):.4f}",
        f"- Independent RMS mean over seeds: {float(multiseed_summary['independent_rms_error_mean_over_seeds']):.4f}",
        f"- Independent P95 mean over seeds: {float(multiseed_summary['independent_p95_error_mean_over_seeds']):.4f}",
        "",
        "## Independent Benchmark (Per-Frequency) Summary",
        f"- RMS mean relative error: {independent_summary['rms_error_mean']:.4f}",
        f"- RMS p90 relative error: {independent_summary['rms_error_p90']:.4f}",
        f"- P95 mean relative error: {independent_summary['p95_error_mean']:.4f}",
        f"- P95 p90 relative error: {independent_summary['p95_error_p90']:.4f}",
        "",
        "## Multi-Seed Rows",
        "",
        "| seed | internal_rms_mean | internal_p95_mean | independent_rms_mean | independent_p95_mean | internal_pass | independent_pass |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in multiseed_rows:
        lines.append(
            "| "
            f"{int(row['seed'])} | "
            f"{float(row['internal_rms_error_mean']):.4f} | "
            f"{float(row['internal_p95_error_mean']):.4f} | "
            f"{float(row['independent_rms_error_mean']):.4f} | "
            f"{float(row['independent_p95_error_mean']):.4f} | "
            f"{'Y' if bool(row['internal_pass_strict']) else 'N'} | "
            f"{'Y' if bool(row['independent_pass_strict']) else 'N'} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = sha.default_config()
    frequencies = _full_integer_grid()
    analysis_config = AnalysisConfig.from_dict({**config.to_dict(), "frequencies_khz": frequencies})
    multiseed_n_samples = max(analysis_config.review_ready_min_samples // 2, 1024)
    seeds = [analysis_config.seed, 11, 23, 37, 53]

    multiseed_rows = run_multiseed_validation_study(
        config=analysis_config,
        seeds=seeds,
        n_samples=multiseed_n_samples,
        frequencies_khz=frequencies,
    )
    multiseed_summary = summarize_multiseed_rows(multiseed_rows)

    independent_n_samples = max(analysis_config.review_ready_min_samples, 2048)
    independent = IndependentBenchmarkRunner(
        analysis_config, seed=analysis_config.seed + 987_654
    ).run(
        n_samples=independent_n_samples,
        frequencies_khz=frequencies,
    )
    analytical = compute_analytical_statistics(analysis_config)
    independent_validation = compare_with_feeder_benchmark(
        analytical=analytical,
        simulated=independent.statistics_frame,
    ).rows
    independent_summary = summarize_validation_errors(independent_validation)

    out_dir = Path("synthetic_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    multiseed_rows_path = out_dir / "multiseed_validation_summary.csv"
    multiseed_aggregate_path = out_dir / "multiseed_validation_aggregate.csv"
    independent_validation_path = out_dir / "independent_benchmark_per_frequency_validation.csv"
    report_path = out_dir / "reviewer_readiness_report.md"

    _write_csv(multiseed_rows_path, multiseed_rows)
    _write_csv(multiseed_aggregate_path, [multiseed_summary])
    _write_csv(independent_validation_path, independent_validation)
    _report_markdown(
        path=report_path,
        multiseed_summary=multiseed_summary,
        independent_summary=independent_summary,
        multiseed_rows=multiseed_rows,
    )

    print(multiseed_rows_path)
    print(multiseed_aggregate_path)
    print(independent_validation_path)
    print(report_path)


if __name__ == "__main__":
    main()
