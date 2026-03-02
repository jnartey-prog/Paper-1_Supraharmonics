from __future__ import annotations

import math
from pathlib import Path

import pytest

from supraharmonic_aggregation.simulation import SyntheticDataGenerator


@pytest.mark.unit
def test_synthetic_data_reproducible_for_fixed_seed(baseline_config) -> None:
    generator_a = SyntheticDataGenerator(baseline_config, seed=101)
    generator_b = SyntheticDataGenerator(baseline_config, seed=101)

    dataset_a = generator_a.generate(n_samples=12)
    dataset_b = generator_b.generate(n_samples=12)

    assert dataset_a.observations == dataset_b.observations
    assert dataset_a.statistics_frame == dataset_b.statistics_frame
    assert dataset_a.validation_frame == dataset_b.validation_frame


@pytest.mark.unit
def test_synthetic_data_shapes_and_validation_columns(baseline_config) -> None:
    n_samples = 8
    dataset = SyntheticDataGenerator(baseline_config, seed=23).generate(n_samples=n_samples)

    expected_observations = n_samples * len(baseline_config.frequencies_khz)
    assert len(dataset.observations) == expected_observations
    assert len(dataset.statistics_frame) == len(baseline_config.frequencies_khz)
    assert len(dataset.validation_frame) == len(baseline_config.frequencies_khz)
    assert all("relative_error_rms" in row for row in dataset.validation_frame)
    assert all("relative_error_p95" in row for row in dataset.validation_frame)


@pytest.mark.unit
def test_synthetic_data_can_omit_complex_columns(baseline_config) -> None:
    dataset = SyntheticDataGenerator(baseline_config, seed=31).generate(
        n_samples=4,
        include_complex=False,
    )
    first_row = dataset.observations[0]
    assert "real_v" not in first_row
    assert "imag_v" not in first_row


@pytest.mark.unit
def test_save_latest_removes_older_prefixed_files(baseline_config, tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "synthetic_latest_old.csv"
    stale.write_text("stale", encoding="utf-8")
    keep = out_dir / "keep.csv"
    keep.write_text("keep", encoding="utf-8")
    note = out_dir / "note.md"
    note.write_text("old", encoding="utf-8")

    generator = SyntheticDataGenerator(baseline_config, seed=55)
    dataset = generator.generate(n_samples=5)
    paths = generator.save_latest(dataset=dataset, output_dir=str(out_dir))

    expected = {
        "synthetic_latest_observations.csv",
        "synthetic_latest_per_frequency_samples.csv",
        "synthetic_latest_statistics.csv",
        "synthetic_latest_analytical.csv",
        "synthetic_latest_validation.csv",
    }
    actual = {path.name for path in out_dir.glob("synthetic_latest_*.csv")}
    assert actual == expected
    assert not stale.exists()
    assert not keep.exists()
    assert not note.exists()
    assert all(Path(path).exists() for path in paths.values())


@pytest.mark.unit
def test_analytical_proxy_mean_abs_is_not_zero_at_zero_coherence(baseline_config) -> None:
    baseline_config.coherence = 0.0
    dataset = SyntheticDataGenerator(baseline_config, seed=19).generate(n_samples=18)
    assert all(float(row["mean_abs_v"]) > 0.0 for row in dataset.analytical_frame)


@pytest.mark.unit
def test_synthetic_cross_frequency_is_not_perfectly_locked(baseline_config) -> None:
    baseline_config.coherence = 0.2
    dataset = SyntheticDataGenerator(baseline_config, seed=77).generate(n_samples=72)
    rows = dataset.observations
    frequencies = sorted({float(row["frequency_khz"]) for row in rows})
    by_sample: dict[int, dict[float, float]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        by_sample.setdefault(sample_id, {})[float(row["frequency_khz"])] = float(row["abs_v"])

    x = [by_sample[sample_id][frequencies[0]] for sample_id in sorted(by_sample)]
    for frequency in frequencies[1:]:
        y = [by_sample[sample_id][frequency] for sample_id in sorted(by_sample)]
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x)
        vx = sum((a - mx) ** 2 for a in x) / len(x)
        vy = sum((b - my) ** 2 for b in y) / len(y)
        corr = cov / math.sqrt(max(vx * vy, 1e-12))
        assert corr < 0.995
