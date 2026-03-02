"""Input/output utilities and schema validation."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from scripts import config


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "baseline_realization": (
        "setting_id",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_shape",
        "region_R_m",
        "n_realizations_per_setting",
        "N",
        "Vagg_real_V",
        "Vagg_imag_V",
        "Vagg_mag_V",
        "denom_mag_min_over_domain",
    ),
    "baseline_setting": (
        "setting_id",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_R_m",
        "mean_Vmag",
        "rms_Vmag",
        "var_Vmag",
        "p95_Vmag",
        "p99_Vmag",
        "exceed_tau",
        "mean_N",
        "var_N",
    ),
    "baseline_scenario": (
        "setting_id",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_shape",
        "region_R_m",
        "n_realizations",
        "tau_V",
        "denom_mag_min_over_domain",
        "seed_setting",
    ),
    "robust_realization": (
        "setting_id",
        "scenario_name",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_shape",
        "region_R_m",
        "n_realizations_per_setting",
        "N",
        "Vagg_real_V",
        "Vagg_imag_V",
        "Vagg_mag_V",
        "denom_mag_min_over_domain",
        "phase_model",
        "current_model",
        "Y_model",
        "kappa",
    ),
    "robust_setting": (
        "setting_id",
        "scenario_name",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_R_m",
        "n_realizations",
        "tau_V",
        "mean_Vmag",
        "rms_Vmag",
        "var_Vmag",
        "p90_Vmag",
        "p95_Vmag",
        "p99_Vmag",
        "exceed_tau",
        "mean_N",
        "var_N",
        "denom_mag_min_over_domain",
        "phase_model",
        "current_model",
        "Y_model",
        "kappa",
        "seed_setting",
    ),
    "robust_scenario": (
        "setting_id",
        "scenario_name",
        "frequency_hz",
        "band_width_hz",
        "lambda_per_m2",
        "region_shape",
        "region_R_m",
        "n_realizations",
        "tau_V",
        "phase_model",
        "current_model",
        "Y_model",
        "kappa",
        "denom_mag_min_over_domain",
        "seed_setting",
    ),
}


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found: {path}")
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_COLUMNS[label], label)
    return df


def _validate_baseline_grid(baseline_setting: pd.DataFrame) -> None:
    freqs = sorted(baseline_setting["frequency_hz"].unique().tolist())
    if len(freqs) < 2:
        raise ValueError("Baseline full-band setting file has fewer than 2 frequency bins.")
    step_set = sorted({round(freqs[i + 1] - freqs[i], 6) for i in range(len(freqs) - 1)})
    if step_set != [2000.0]:
        raise ValueError(
            "Baseline frequency grid is not strictly 2 kHz stepped. "
            f"Observed unique steps: {step_set}"
        )
    if abs(freqs[0] - 2000.0) > 1e-9 or abs(freqs[-1] - 150000.0) > 1e-9:
        raise ValueError(
            "Baseline frequency range is not 2-150 kHz exactly. "
            f"Observed min/max: {freqs[0]}, {freqs[-1]}"
        )


def ensure_output_dirs(paths: config.Paths) -> None:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def clear_previous_outputs(paths: config.Paths) -> None:
    for pattern in ("*.csv", "*.md"):
        for p in paths.tables_dir.glob(pattern):
            p.unlink(missing_ok=True)
    for pattern in ("*.png", "*.csv"):
        for p in paths.figures_dir.glob(pattern):
            p.unlink(missing_ok=True)
    # Remove stale placeholder artifacts from older runs.
    artifacts_dir = paths.root / "manuscript" / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir, ignore_errors=True)


def load_inputs(paths: config.Paths) -> dict[str, Any]:
    baseline_realization = _load_csv(paths.baseline_realization, "baseline_realization")
    baseline_setting = _load_csv(paths.baseline_setting, "baseline_setting")
    baseline_scenario = _load_csv(paths.baseline_scenario, "baseline_scenario")
    robust_realization = _load_csv(paths.robust_realization, "robust_realization")
    robust_setting = _load_csv(paths.robust_setting, "robust_setting")
    robust_scenario = _load_csv(paths.robust_scenario, "robust_scenario")

    _validate_baseline_grid(baseline_setting)

    outline_text = paths.outline.read_text(encoding="utf-8")
    concept_text = paths.concept.read_text(encoding="utf-8")

    return {
        "baseline_realization": baseline_realization,
        "baseline_setting": baseline_setting,
        "baseline_scenario": baseline_scenario,
        "robust_realization": robust_realization,
        "robust_setting": robust_setting,
        "robust_scenario": robust_scenario,
        "outline_text": outline_text,
        "concept_text": concept_text,
    }


def nearest_value(values: list[float], target: float) -> float:
    if not values:
        raise ValueError("Cannot find nearest value in an empty list.")
    return min(values, key=lambda v: abs(v - target))
