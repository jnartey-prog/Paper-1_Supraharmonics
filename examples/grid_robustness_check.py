"""Run frequency-grid robustness checks for synthetic validation quality."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import supraharmonic_aggregation as sha


def _grid_even_2khz() -> list[float]:
    return [float(value) for value in range(2, 151, 2)]


def _grid_odd_offset_2khz() -> list[float]:
    return [float(value) for value in range(3, 150, 2)]


def _grid_mixed_dense_resonance() -> list[float]:
    values = set(range(2, 151, 4))
    values.update(range(18, 43, 1))
    values.update(range(68, 93, 1))
    values.update(range(118, 151, 2))
    return [float(value) for value in sorted(values)]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


def _summarize_validation(
    name: str, frame: list[dict[str, float | int | str]]
) -> dict[str, float | str]:
    rms = [float(row["relative_error_rms"]) for row in frame]
    p95 = [float(row["relative_error_p95"]) for row in frame]
    return {
        "grid_name": name,
        "n_frequencies": float(len(frame)),
        "rms_error_mean": statistics.fmean(rms),
        "rms_error_median": statistics.median(rms),
        "rms_error_p90": _quantile(rms, 0.90),
        "rms_error_max": max(rms),
        "p95_error_mean": statistics.fmean(p95),
        "p95_error_median": statistics.median(p95),
        "p95_error_p90": _quantile(p95, 0.90),
        "p95_error_max": max(p95),
    }


def _compare_against_baseline(
    rows: list[dict[str, float | str]],
    baseline_name: str,
) -> list[dict[str, float | str]]:
    baseline = next(row for row in rows if row["grid_name"] == baseline_name)
    out: list[dict[str, float | str]] = []
    for row in rows:
        result = dict(row)
        if row["grid_name"] == baseline_name:
            result["delta_rms_error_mean_pct_vs_even"] = 0.0
            result["delta_p95_error_mean_pct_vs_even"] = 0.0
        else:
            result["delta_rms_error_mean_pct_vs_even"] = (
                100.0
                * (float(row["rms_error_mean"]) - float(baseline["rms_error_mean"]))
                / max(float(baseline["rms_error_mean"]), 1e-12)
            )
            result["delta_p95_error_mean_pct_vs_even"] = (
                100.0
                * (float(row["p95_error_mean"]) - float(baseline["p95_error_mean"]))
                / max(float(baseline["p95_error_mean"]), 1e-12)
            )
        result["passes_strict_gate"] = (
            float(result["rms_error_p90"]) <= 0.10 and float(result["p95_error_p90"]) <= 0.10
        )
        out.append(result)
    return out


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Frequency Grid Robustness Summary",
        "",
        "All grids pass strict gates (p90 RMS and p90 P95 relative error <= 0.10) if `passes_strict_gate=True`.",
        "",
        "| Grid | n_freq | RMS mean | RMS p90 | P95 mean | P95 p90 | dRMS mean % vs even | dP95 mean % vs even | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['grid_name']} | "
            f"{int(float(row['n_frequencies']))} | "
            f"{float(row['rms_error_mean']):.4f} | "
            f"{float(row['rms_error_p90']):.4f} | "
            f"{float(row['p95_error_mean']):.4f} | "
            f"{float(row['p95_error_p90']):.4f} | "
            f"{float(row['delta_rms_error_mean_pct_vs_even']):.2f} | "
            f"{float(row['delta_p95_error_mean_pct_vs_even']):.2f} | "
            f"{'Y' if row['passes_strict_gate'] else 'N'} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = sha.default_config()
    generator = sha.SyntheticDataGenerator(config, seed=config.seed)
    sample_count = max(config.review_ready_min_samples, config.monte_carlo_samples)

    grids = {
        "even_2khz": _grid_even_2khz(),
        "odd_offset_2khz": _grid_odd_offset_2khz(),
        "mixed_dense_resonance": _grid_mixed_dense_resonance(),
    }

    summaries: list[dict[str, float | str]] = []
    for name, frequencies in grids.items():
        dataset = generator.generate(
            n_samples=sample_count,
            include_complex=False,
            frequencies_khz=frequencies,
            include_measurement_noise=True,
        )
        summaries.append(_summarize_validation(name=name, frame=dataset.validation_frame))

    compared = _compare_against_baseline(summaries, baseline_name="even_2khz")
    out_dir = Path("synthetic_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frequency_grid_robustness_summary.csv"
    md_path = out_dir / "frequency_grid_robustness_summary.md"
    _write_csv(csv_path, compared)
    _write_markdown(md_path, compared)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
