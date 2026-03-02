"""Pipeline configuration and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from pathlib import Path


REQUESTED_ROOT = Path("/mnt/data")
FALLBACK_ROOT = Path(__file__).resolve().parents[1]


BASELINE_REALIZATION_FILE = "synthetic_supraharmonic_realization_fullband_2kHz.csv"
BASELINE_SETTING_FILE = "synthetic_supraharmonic_setting_summary_fullband_2kHz.csv"
BASELINE_SCENARIO_FILE = "synthetic_scenario_matrix_fullband_2kHz.csv"

ROBUST_REALIZATION_FILE = "synthetic_supraharmonic_realization_robust_subsetfreq.csv"
ROBUST_SETTING_FILE = "synthetic_supraharmonic_setting_summary_robust_subsetfreq.csv"
ROBUST_SCENARIO_FILE = "synthetic_scenario_matrix_robust_subsetfreq.csv"

OUTLINE_FILE = "outline.txt"
CONCEPT_FILE = "concept.txt"

FEEDER_BENCHMARK_KERNELS_FILE = "feeder_benchmark_kernels.csv"
FEEDER_BENCHMARK_FEATURES_FILE = "feeder_benchmark_kernel_features.csv"
FEEDER_MODEL_REALIZATIONS_FILE = "feeder_model_simulated_realizations.csv"
FEEDER_MODEL_SETTING_SUMMARY_FILE = "feeder_model_setting_summary.csv"


BOOTSTRAP_RESAMPLES = 500
CI_LEVEL = 0.95
RNG_SEED = 20260220

PLOT_DPI = 600

GATE_POISSON_TOL = 0.15
GATE_CANCELLATION_FACTOR = 0.02
GATE_DENOM_MIN = 0.05

BASELINE_TARGET_LAMBDA = 2e-5
BASELINE_TARGET_R = 500.0
BASELINE_TARGET_FREQ_FIG2_KHZ = (10.0, 30.0, 100.0)
BASELINE_TARGET_FREQ_FIG3_HZ = 20000.0
ROBUST_TARGET_FREQ_HZ = 20000.0
ROBUST_TARGET_SCENARIOS = {
    "baseline": "baseline_ppp",
    "clustered": "clustered_thomas",
    "coherence": "partial_coherence_kappa_10",
    "inhomogeneous": "inhomog_hotspot",
}
ALLOWABLE_LAMBDA_P99_LIMIT_V = 0.30

FREQ_MIN_HZ = 2000.0
FREQ_MAX_HZ = 150000.0
FREQ_STEP_HZ = 2000.0

DIST_MIN_M = 0.0
DIST_MAX_M = 1000.0
DIST_STEP_M = 10.0
DIST_ATTENUATION_D0_M = 50.0

FEEDER_SIM_REALIZATIONS_PER_SETTING = 200
Y_C_EQ_F = 2.0e-6
I_LOGN_MU_LN = -3.65
I_LOGN_SIGMA_LN = 0.55


@dataclass(frozen=True)
class FeederSpec:
    feeder_id: str
    feeder_name: str
    length_km: float
    R_ohm_per_km: float
    L_H_per_km: float
    C_F_per_km: float
    R_th_ohm: float
    L_th_H: float
    load_impedance_ohm: float = 0.30
    G_shunt_S: float = 0.0
    skin_effect_coeff: float = 0.18
    dielectric_tan_delta: float = 0.015


FEEDER_SPECS: Dict[str, FeederSpec] = {
    "A": FeederSpec(
        feeder_id="A",
        feeder_name="urban_short",
        length_km=0.5,
        R_ohm_per_km=0.24,
        L_H_per_km=0.38e-3,
        C_F_per_km=190e-9,
        R_th_ohm=0.010,
        L_th_H=65e-6,
        load_impedance_ohm=0.28,
        G_shunt_S=0.0,
        skin_effect_coeff=0.14,
        dielectric_tan_delta=0.012,
    ),
    "B": FeederSpec(
        feeder_id="B",
        feeder_name="suburban_medium",
        length_km=2.0,
        R_ohm_per_km=0.38,
        L_H_per_km=0.52e-3,
        C_F_per_km=120e-9,
        R_th_ohm=0.014,
        L_th_H=105e-6,
        load_impedance_ohm=0.31,
        G_shunt_S=0.0,
        skin_effect_coeff=0.20,
        dielectric_tan_delta=0.016,
    ),
    "C": FeederSpec(
        feeder_id="C",
        feeder_name="rural_long",
        length_km=5.0,
        R_ohm_per_km=0.62,
        L_H_per_km=0.78e-3,
        C_F_per_km=82e-9,
        R_th_ohm=0.020,
        L_th_H=150e-6,
        load_impedance_ohm=0.35,
        G_shunt_S=0.0,
        skin_effect_coeff=0.28,
        dielectric_tan_delta=0.022,
    ),
}


@dataclass(frozen=True)
class Paths:
    root: Path
    baseline_realization: Path
    baseline_setting: Path
    baseline_scenario: Path
    robust_realization: Path
    robust_setting: Path
    robust_scenario: Path
    outline: Path
    concept: Path
    feeder_benchmark_kernels: Path
    feeder_benchmark_features: Path
    feeder_model_realizations: Path
    feeder_model_setting_summary: Path
    tables_dir: Path
    figures_dir: Path


def _has_required_files(root: Path) -> bool:
    required = [
        BASELINE_REALIZATION_FILE,
        BASELINE_SETTING_FILE,
        BASELINE_SCENARIO_FILE,
        ROBUST_REALIZATION_FILE,
        ROBUST_SETTING_FILE,
        ROBUST_SCENARIO_FILE,
        OUTLINE_FILE,
        CONCEPT_FILE,
    ]
    return all((root / name).exists() for name in required)


def resolve_root() -> Path:
    if _has_required_files(REQUESTED_ROOT):
        return REQUESTED_ROOT
    if _has_required_files(FALLBACK_ROOT):
        return FALLBACK_ROOT
    missing = [
        name
        for name in [
            BASELINE_REALIZATION_FILE,
            BASELINE_SETTING_FILE,
            BASELINE_SCENARIO_FILE,
            ROBUST_REALIZATION_FILE,
            ROBUST_SETTING_FILE,
            ROBUST_SCENARIO_FILE,
            OUTLINE_FILE,
            CONCEPT_FILE,
        ]
        if not (REQUESTED_ROOT / name).exists() and not (FALLBACK_ROOT / name).exists()
    ]
    raise FileNotFoundError(
        "Could not locate required input files under either "
        f"{REQUESTED_ROOT} or {FALLBACK_ROOT}. Missing: {missing}"
    )


def build_paths() -> Paths:
    root = resolve_root()
    tables_dir = root / "manuscript" / "tables"
    figures_dir = root / "manuscript" / "figures"
    return Paths(
        root=root,
        baseline_realization=root / BASELINE_REALIZATION_FILE,
        baseline_setting=root / BASELINE_SETTING_FILE,
        baseline_scenario=root / BASELINE_SCENARIO_FILE,
        robust_realization=root / ROBUST_REALIZATION_FILE,
        robust_setting=root / ROBUST_SETTING_FILE,
        robust_scenario=root / ROBUST_SCENARIO_FILE,
        outline=root / OUTLINE_FILE,
        concept=root / CONCEPT_FILE,
        feeder_benchmark_kernels=root / FEEDER_BENCHMARK_KERNELS_FILE,
        feeder_benchmark_features=root / FEEDER_BENCHMARK_FEATURES_FILE,
        feeder_model_realizations=root / FEEDER_MODEL_REALIZATIONS_FILE,
        feeder_model_setting_summary=root / FEEDER_MODEL_SETTING_SUMMARY_FILE,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
