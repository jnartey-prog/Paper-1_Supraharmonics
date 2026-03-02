"""Table generation for manuscript outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from scripts import config, metrics


def _write_table(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return int(df.shape[0])


def _write_text(text: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return int(len(text.splitlines()))


def _read_required_csv(path: Path, required_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df


def _freq_grid_description(freqs: np.ndarray) -> str:
    unique = np.sort(np.unique(freqs))
    if unique.size == 0:
        return ""
    if unique.size == 1:
        return f"{unique[0]:.1f} (n=1)"
    step = unique[1] - unique[0]
    return f"{unique.min():.1f}:{step:.1f}:{unique.max():.1f} (n={unique.size})"


def _dataset_meta(df: pd.DataFrame, dataset_tag: str) -> dict[str, Any]:
    freqs = np.sort(df["frequency_hz"].unique())
    return {
        "dataset_tag": dataset_tag,
        "metadata_k": int(df.filter(regex="n_realizations").iloc[:, 0].median()),
        "metadata_bin_width_hz": float(df["band_width_hz"].median()),
        "metadata_frequency_range_hz": f"{freqs.min():.0f}-{freqs.max():.0f}",
    }


def table1_scenario_matrix(
    baseline_scenario: pd.DataFrame, robust_scenario: pd.DataFrame, out_dir: Path
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []

    bmeta = _dataset_meta(baseline_scenario, "baseline_fullband_2kHz")
    rows.append(
        {
            "scenario_name": "baseline_ppp",
            "f_grid_description_hz": _freq_grid_description(
                baseline_scenario["frequency_hz"].to_numpy()
            ),
            "lambda_levels_per_m2": ",".join(
                f"{v:g}" for v in np.sort(baseline_scenario["lambda_per_m2"].unique())
            ),
            "R_levels_m": ",".join(
                f"{v:g}" for v in np.sort(baseline_scenario["region_R_m"].unique())
            ),
            "K": int(baseline_scenario["n_realizations"].median()),
            "tau_V": float(baseline_scenario["tau_V"].median()),
            "bin_width_hz": float(baseline_scenario["band_width_hz"].median()),
            "seed_setting": f"{int(baseline_scenario['seed_setting'].min())}-{int(baseline_scenario['seed_setting'].max())}",
            "scenario_definition": "baseline PPP synthetic full-band grid",
            **bmeta,
        }
    )

    rmeta = _dataset_meta(robust_scenario, "robust_subsetfreq")
    for scen, grp in robust_scenario.groupby("scenario_name", sort=True):
        phase_model = ",".join(sorted(grp["phase_model"].dropna().astype(str).unique()))
        current_model = ",".join(sorted(grp["current_model"].dropna().astype(str).unique()))
        y_model = ",".join(sorted(grp["Y_model"].dropna().astype(str).unique()))
        kappa_vals = grp["kappa"].dropna().unique()
        kappa_txt = (
            "none" if len(kappa_vals) == 0 else ",".join(f"{v:g}" for v in np.sort(kappa_vals))
        )
        rows.append(
            {
                "scenario_name": scen,
                "f_grid_description_hz": _freq_grid_description(grp["frequency_hz"].to_numpy()),
                "lambda_levels_per_m2": ",".join(
                    f"{v:g}" for v in np.sort(grp["lambda_per_m2"].unique())
                ),
                "R_levels_m": ",".join(f"{v:g}" for v in np.sort(grp["region_R_m"].unique())),
                "K": int(grp["n_realizations"].median()),
                "tau_V": float(grp["tau_V"].median()),
                "bin_width_hz": float(grp["band_width_hz"].median()),
                "seed_setting": f"{int(grp['seed_setting'].min())}-{int(grp['seed_setting'].max())}",
                "scenario_definition": f"phase={phase_model}; current={current_model}; Y={y_model}; kappa={kappa_txt}",
                **rmeta,
            }
        )

    df = pd.DataFrame(rows)
    path = out_dir / "table1_scenario_matrix.csv"
    return path, _write_table(df, path)


def table2_analytical_summary(concept_text: str, out_dir: Path) -> tuple[Path, int]:
    # Concise summary aligned with provided concept notation/assumptions.
    rows = [
        {
            "deliverable": "Target aggregate quantity",
            "notation": "Vagg(f) in C",
            "expression_summary": "Complex aggregate PCC voltage phasor per frequency bin.",
            "assumption_or_condition": "Frequency-domain, per-bin analysis in 2-150 kHz.",
        },
        {
            "deliverable": "Source emission mark",
            "notation": "I_i(f)=I_sh,i(f) exp(j phi_i(f))",
            "expression_summary": "Norton current magnitude-phase mark per source.",
            "assumption_or_condition": "IID marks and asynchronous phases in baseline.",
        },
        {
            "deliverable": "Effective injection with absorption",
            "notation": "I_tilde_i(f)=I_sh,i exp(j phi_i)/(1+Y(f) Z_tr(f,d_i))",
            "expression_summary": "Source contribution reduced by internal admittance and transfer impedance.",
            "assumption_or_condition": "Denominator separation condition required.",
        },
        {
            "deliverable": "Aggregate shot-noise form",
            "notation": "V(f)=sum_{x_i in Phi} g(f,x_i,m_i)",
            "expression_summary": "Marked spatial shot-noise aggregation.",
            "assumption_or_condition": "PPP baseline deployment.",
        },
        {
            "deliverable": "Integrability/boundedness",
            "notation": "|Z_tr(f,d)| <= C_f a_f(d), |1+Y Z_tr| >= c0",
            "expression_summary": "Finite moments via attenuation envelope and denominator lower bound.",
            "assumption_or_condition": "A8-style conditions in concept text.",
        },
        {
            "deliverable": "Scaling note",
            "notation": "RMS approx sqrt(lambda) (incoherent baseline)",
            "expression_summary": "Incoherent limit produces sublinear growth with intensity.",
            "assumption_or_condition": "Phase cancellation baseline.",
        },
        {
            "deliverable": "Coherence factor note",
            "notation": "E[exp(j phi)] = rho exp(j psi)",
            "expression_summary": "Residual mean appears when rho > 0 (partial coherence).",
            "assumption_or_condition": "Robustness departure from uniform phase.",
        },
    ]
    df = pd.DataFrame(rows)
    df["dataset_tag"] = "concept_text"
    df["metadata_note_k"] = "not_applicable_theory_summary"
    df["metadata_note_bin_width"] = "not_applicable_theory_summary"
    df["metadata_frequency_range_note_hz"] = "2000-150000"
    path = out_dir / "table2_analytical_summary.csv"
    return path, _write_table(df, path)


def table3_baseline_results_by_frequency(
    baseline_realization: pd.DataFrame,
    baseline_scenario: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, int]:
    tau_v = float(baseline_scenario["tau_V"].median())
    subset = baseline_realization[
        (np.isclose(baseline_realization["lambda_per_m2"], config.BASELINE_TARGET_LAMBDA))
        & (np.isclose(baseline_realization["region_R_m"], config.BASELINE_TARGET_R))
    ].copy()
    if subset.empty:
        raise ValueError("No baseline rows found for lambda=2e-5 and R=500 m.")

    rows: list[dict[str, Any]] = []
    for i, (freq, grp) in enumerate(subset.groupby("frequency_hz", sort=True)):
        mags = grp["Vagg_mag_V"].to_numpy(dtype=float)
        m = metrics.metrics_with_ci(
            mags=mags,
            tau_v=tau_v,
            n_resamples=config.BOOTSTRAP_RESAMPLES,
            ci_level=config.CI_LEVEL,
            seed=config.RNG_SEED + i * 97,
        )
        rows.append(
            {
                "frequency_hz": float(freq),
                "n_realizations": int(grp.shape[0]),
                "mean_vmag": m["mean"],
                "mean_ci_lo": m["mean_ci_lo"],
                "mean_ci_hi": m["mean_ci_hi"],
                "rms_vmag": m["rms"],
                "rms_ci_lo": m["rms_ci_lo"],
                "rms_ci_hi": m["rms_ci_hi"],
                "p90_vmag": m["p90"],
                "p95_vmag": m["p95"],
                "p95_ci_lo": m["p95_ci_lo"],
                "p95_ci_hi": m["p95_ci_hi"],
                "p99_vmag": m["p99"],
                "p99_ci_lo": m["p99_ci_lo"],
                "p99_ci_hi": m["p99_ci_hi"],
                "exceedance_tau": m["exceedance"],
                "exceedance_ci_lo": m["exceedance_ci_lo"],
                "exceedance_ci_hi": m["exceedance_ci_hi"],
                "tau_v": tau_v,
                "dataset_tag": "baseline_fullband_2kHz",
                "metadata_k": int(grp["n_realizations_per_setting"].median()),
                "metadata_bin_width_hz": float(grp["band_width_hz"].median()),
                "metadata_frequency_range_hz": "2000-150000",
            }
        )
    df = pd.DataFrame(rows).sort_values("frequency_hz")
    path = out_dir / "table3_baseline_ppp_results_by_frequency.csv"
    return path, _write_table(df, path)


def table4_robustness_effect_sizes(robust_setting: pd.DataFrame, out_dir: Path) -> tuple[Path, int]:
    key_cols = ["frequency_hz", "lambda_per_m2", "region_R_m"]
    base = robust_setting[
        robust_setting["scenario_name"] == config.ROBUST_TARGET_SCENARIOS["baseline"]
    ].copy()
    if base.empty:
        raise ValueError("Robust setting summary missing baseline_ppp scenario.")

    rows: list[pd.DataFrame] = []
    for scen in sorted(robust_setting["scenario_name"].unique()):
        if scen == config.ROBUST_TARGET_SCENARIOS["baseline"]:
            continue
        scen_df = robust_setting[robust_setting["scenario_name"] == scen].copy()
        merged = scen_df.merge(base, on=key_cols, suffixes=("_scen", "_base"), how="inner")
        if merged.empty:
            continue
        for metric in ("mean_Vmag", "var_Vmag", "rms_Vmag", "p99_Vmag", "exceed_tau"):
            merged[f"delta_{metric}"] = merged[f"{metric}_scen"] - merged[f"{metric}_base"]
            denom = merged[f"{metric}_base"].replace(0.0, np.nan)
            merged[f"delta_pct_{metric}"] = 100.0 * merged[f"delta_{metric}"] / denom
        merged["scenario_name"] = scen
        rows.append(merged)

    if not rows:
        raise ValueError("No robustness scenarios found to compare against baseline_ppp.")

    out = pd.concat(rows, ignore_index=True)
    scenario_scores = (
        out.groupby("scenario_name")[
            [
                "delta_pct_mean_Vmag",
                "delta_pct_rms_Vmag",
                "delta_pct_p99_Vmag",
                "delta_pct_exceed_tau",
            ]
        ]
        .apply(lambda d: d.abs().median().mean())
        .sort_values(ascending=False)
    )
    ranks = {name: idx + 1 for idx, name in enumerate(scenario_scores.index.tolist())}
    out["dominant_factor_rank"] = out["scenario_name"].map(ranks)
    out["dataset_tag"] = "robust_subsetfreq"
    out["metadata_k"] = out["n_realizations_scen"].astype(int)
    out["metadata_bin_width_hz"] = out["band_width_hz_scen"]
    out["metadata_frequency_range_hz"] = "10000-150000"

    keep_cols = [
        "scenario_name",
        "frequency_hz",
        "lambda_per_m2",
        "region_R_m",
        "delta_mean_Vmag",
        "delta_pct_mean_Vmag",
        "delta_var_Vmag",
        "delta_pct_var_Vmag",
        "delta_rms_Vmag",
        "delta_pct_rms_Vmag",
        "delta_p99_Vmag",
        "delta_pct_p99_Vmag",
        "delta_exceed_tau",
        "delta_pct_exceed_tau",
        "dominant_factor_rank",
        "dataset_tag",
        "metadata_k",
        "metadata_bin_width_hz",
        "metadata_frequency_range_hz",
    ]
    df = out[keep_cols].sort_values(
        ["scenario_name", "frequency_hz", "lambda_per_m2", "region_R_m"]
    )
    path = out_dir / "table4_robustness_effect_sizes.csv"
    return path, _write_table(df, path)


def table5_feeder_model_kernel_features(paths: config.Paths, out_dir: Path) -> tuple[Path, int]:
    features = _read_required_csv(
        paths.feeder_benchmark_features,
        [
            "feeder_id",
            "feeder_name",
            "frequency_hz",
            "band_width_hz",
            "Zmag_0m",
            "Zmag_100m",
            "Zmag_500m",
            "Zmag_1000m",
            "fitted_alpha",
            "resonance_index",
        ],
    )
    spec_rows = []
    for feeder_id in sorted(config.FEEDER_SPECS.keys()):
        spec = config.FEEDER_SPECS[feeder_id]
        spec_rows.append(
            {
                "feeder_id": spec.feeder_id,
                "feeder_name": spec.feeder_name,
                "feeder_length_km": spec.length_km,
                "R_ohm_per_km": spec.R_ohm_per_km,
                "L_H_per_km": spec.L_H_per_km,
                "C_F_per_km": spec.C_F_per_km,
                "R_th_ohm": spec.R_th_ohm,
                "L_th_H": spec.L_th_H,
                "G_shunt_S": spec.G_shunt_S,
            }
        )
    specs = pd.DataFrame(spec_rows)
    df = features.merge(specs, on=["feeder_id", "feeder_name"], how="left")
    df["benchmark_label"] = "feeder-model benchmark"
    df = df[
        [
            "benchmark_label",
            "feeder_id",
            "feeder_name",
            "feeder_length_km",
            "R_ohm_per_km",
            "L_H_per_km",
            "C_F_per_km",
            "R_th_ohm",
            "L_th_H",
            "G_shunt_S",
            "frequency_hz",
            "band_width_hz",
            "Zmag_0m",
            "Zmag_100m",
            "Zmag_500m",
            "Zmag_1000m",
            "fitted_alpha",
            "resonance_index",
        ]
    ].sort_values(["feeder_id", "frequency_hz"])
    path = out_dir / "table5_feeder_model_kernel_features.csv"
    return path, _write_table(df, path)


def _kernel_lookup_for_table6(
    kernels: pd.DataFrame,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    lookup: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (fid, freq), grp in kernels.groupby(["feeder_id", "frequency_hz"], sort=False):
        g = grp.sort_values("distance_m")
        d = g["distance_m"].to_numpy(dtype=float)
        z = g["Ztr_real_ohm"].to_numpy(dtype=float) + 1j * g["Ztr_imag_ohm"].to_numpy(dtype=float)
        lookup[(str(fid), int(round(float(freq))))] = (d, z)
    return lookup


def _poisson_pmf_truncated(mu: float) -> tuple[np.ndarray, np.ndarray]:
    if mu <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n_max = max(8, int(np.ceil(mu + 10.0 * np.sqrt(mu + 1.0) + 50.0)))
    pmf = np.zeros(n_max + 1, dtype=float)
    pmf[0] = np.exp(-mu)
    for n in range(1, n_max + 1):
        pmf[n] = pmf[n - 1] * mu / float(n)
    tail = max(0.0, 1.0 - float(np.sum(pmf)))
    if n_max >= 1:
        pmf[n_max] += tail
    n_arr = np.arange(1, n_max + 1, dtype=float)
    return n_arr, pmf[1:]


def _poisson_rayleigh_mixture_exceed(
    v: float,
    n_arr: np.ndarray,
    pmf_n: np.ndarray,
    s2_per_source: float,
) -> float:
    if s2_per_source <= 0 or n_arr.size == 0:
        return 0.0
    expo = -(v * v) / np.maximum(n_arr * s2_per_source, 1e-18)
    return float(np.sum(pmf_n * np.exp(expo)))


def _poisson_rayleigh_mixture_p99(
    n_arr: np.ndarray,
    pmf_n: np.ndarray,
    s2_per_source: float,
    rms_pred: float,
) -> float:
    if s2_per_source <= 0 or n_arr.size == 0:
        return 0.0
    target = 0.01
    lo = 0.0
    hi = max(1.0, 5.0 * max(rms_pred, 1e-6))
    for _ in range(30):
        if _poisson_rayleigh_mixture_exceed(hi, n_arr, pmf_n, s2_per_source) <= target:
            break
        hi *= 1.6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _poisson_rayleigh_mixture_exceed(mid, n_arr, pmf_n, s2_per_source) > target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def table6_analytical_vs_feeder_model_metrics(
    paths: config.Paths, out_dir: Path
) -> tuple[Path, int]:
    kernels = _read_required_csv(
        paths.feeder_benchmark_kernels,
        ["feeder_id", "frequency_hz", "distance_m", "Ztr_real_ohm", "Ztr_imag_ohm"],
    )
    sim = _read_required_csv(
        paths.feeder_model_setting_summary,
        [
            "feeder_id",
            "feeder_name",
            "setting_id",
            "frequency_hz",
            "lambda_per_m2",
            "region_R_m",
            "tau_V",
            "rms_ci_lo",
            "rms_ci_hi",
            "rms_Vmag",
            "p99_ci_lo",
            "p99_ci_hi",
            "p99_Vmag",
            "exceed_ci_lo",
            "exceed_ci_hi",
            "exceed_tau",
        ],
    )
    lookup = _kernel_lookup_for_table6(kernels)
    e_i2 = float(np.exp(2.0 * config.I_LOGN_MU_LN + 2.0 * (config.I_LOGN_SIGMA_LN**2)))

    rows: list[dict[str, Any]] = []
    for r in sim.itertuples(index=False):
        fid = str(r.feeder_id)
        freq_hz = float(r.frequency_hz)
        key = (fid, int(round(freq_hz)))
        if key not in lookup:
            raise KeyError(f"Kernel lookup missing for {key}")
        d_grid, ztr = lookup[key]
        rmax = float(r.region_R_m)
        mask = d_grid <= rmax + 1e-12
        if not np.any(mask):
            raise ValueError(f"No kernel distances <= R={rmax} for feeder={fid}, f={freq_hz}")
        d = d_grid[mask]
        z = ztr[mask]
        y_f = 1j * 2.0 * np.pi * freq_hz * config.Y_C_EQ_F
        h = z / (1.0 + y_f * z)
        integrand = (np.abs(h) ** 2) * (2.0 * np.pi * d)
        e_v2 = float(r.lambda_per_m2) * e_i2 * float(np.trapezoid(integrand, d))
        e_v2 = max(e_v2, 0.0)
        rms_pred = float(np.sqrt(e_v2))

        mu_n = float(r.lambda_per_m2) * np.pi * (rmax**2)
        if mu_n > 0:
            s2_per_source = e_v2 / max(mu_n, 1e-18)
            n_arr, pmf_n = _poisson_pmf_truncated(mu_n)
            p99_pred = _poisson_rayleigh_mixture_p99(
                n_arr=n_arr,
                pmf_n=pmf_n,
                s2_per_source=s2_per_source,
                rms_pred=rms_pred,
            )
            exceed_pred = _poisson_rayleigh_mixture_exceed(
                v=float(r.tau_V),
                n_arr=n_arr,
                pmf_n=pmf_n,
                s2_per_source=s2_per_source,
            )
        else:
            p99_pred = 0.0
            exceed_pred = 0.0

        rms_sim = float(r.rms_Vmag)
        p99_sim = float(r.p99_Vmag)
        exc_sim = float(r.exceed_tau)
        rows.append(
            {
                "benchmark_label": "feeder-model benchmark",
                "surrogate_label": "analytical surrogate (Poisson-Rayleigh mixture tail proxy)",
                "feeder_id": fid,
                "feeder_name": str(r.feeder_name),
                "setting_id": int(r.setting_id),
                "frequency_hz": freq_hz,
                "lambda_per_m2": float(r.lambda_per_m2),
                "region_R_m": rmax,
                "tau_V": float(r.tau_V),
                "rms_pred": rms_pred,
                "rms_sim": rms_sim,
                "rms_sim_ci_lo": float(r.rms_ci_lo),
                "rms_sim_ci_hi": float(r.rms_ci_hi),
                "p99_pred": p99_pred,
                "p99_sim": p99_sim,
                "p99_sim_ci_lo": float(r.p99_ci_lo),
                "p99_sim_ci_hi": float(r.p99_ci_hi),
                "exceed_pred": exceed_pred,
                "exceed_sim": exc_sim,
                "exceed_sim_ci_lo": float(r.exceed_ci_lo),
                "exceed_sim_ci_hi": float(r.exceed_ci_hi),
                "rel_err_RMS": (rms_pred - rms_sim) / max(rms_sim, 1e-12),
                "rel_err_P99": (p99_pred - p99_sim) / max(p99_sim, 1e-12),
                "abs_err_exceed": abs(exceed_pred - exc_sim),
                "abs_err_rms": abs(rms_pred - rms_sim),
                "abs_err_p99": abs(p99_pred - p99_sim),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["feeder_id", "frequency_hz", "lambda_per_m2", "region_R_m"]
    )
    path = out_dir / "table6_analytical_vs_feeder_model_metrics.csv"
    return path, _write_table(df, path)


def tableS6_feeder_model_comparison_summary(paths: config.Paths, out_dir: Path) -> tuple[Path, int]:
    t6 = _read_required_csv(
        out_dir / "table6_analytical_vs_feeder_model_metrics.csv",
        [
            "benchmark_label",
            "surrogate_label",
            "feeder_id",
            "feeder_name",
            "rms_pred",
            "rms_sim",
            "p99_pred",
            "p99_sim",
            "exceed_pred",
            "exceed_sim",
            "rel_err_RMS",
            "rel_err_P99",
            "abs_err_exceed",
            "abs_err_rms",
            "abs_err_p99",
        ],
    )
    rows: list[dict[str, Any]] = []
    for (fid, fname), grp in t6.groupby(["feeder_id", "feeder_name"], sort=True):
        corr_rms = float(np.corrcoef(grp["rms_pred"], grp["rms_sim"])[0, 1])
        corr_p99 = float(np.corrcoef(grp["p99_pred"], grp["p99_sim"])[0, 1])
        rows.append(
            {
                "benchmark_label": "feeder-model benchmark",
                "surrogate_label": str(grp["surrogate_label"].iloc[0]),
                "feeder_id": fid,
                "feeder_name": fname,
                "n_settings": int(grp.shape[0]),
                "mean_rel_err_RMS": float(np.mean(grp["rel_err_RMS"])),
                "mean_rel_err_P99": float(np.mean(grp["rel_err_P99"])),
                "mean_abs_err_exceed": float(np.mean(grp["abs_err_exceed"])),
                "MAE_rms": float(np.mean(np.abs(grp["rms_pred"] - grp["rms_sim"]))),
                "MAE_p99": float(np.mean(np.abs(grp["p99_pred"] - grp["p99_sim"]))),
                "corr_rms": corr_rms,
                "corr_p99": corr_p99,
            }
        )
    all_grp = t6
    rows.append(
        {
            "benchmark_label": "feeder-model benchmark",
            "surrogate_label": str(t6["surrogate_label"].iloc[0]),
            "feeder_id": "all",
            "feeder_name": "all",
            "n_settings": int(all_grp.shape[0]),
            "mean_rel_err_RMS": float(np.mean(all_grp["rel_err_RMS"])),
            "mean_rel_err_P99": float(np.mean(all_grp["rel_err_P99"])),
            "mean_abs_err_exceed": float(np.mean(all_grp["abs_err_exceed"])),
            "MAE_rms": float(np.mean(np.abs(all_grp["rms_pred"] - all_grp["rms_sim"]))),
            "MAE_p99": float(np.mean(np.abs(all_grp["p99_pred"] - all_grp["p99_sim"]))),
            "corr_rms": float(np.corrcoef(all_grp["rms_pred"], all_grp["rms_sim"])[0, 1]),
            "corr_p99": float(np.corrcoef(all_grp["p99_pred"], all_grp["p99_sim"])[0, 1]),
        }
    )
    df = pd.DataFrame(rows)
    path = out_dir / "tableS6_feeder_model_comparison_summary.csv"
    return path, _write_table(df, path)


def tableS1_sanity_checks(
    baseline_realization: pd.DataFrame,
    baseline_scenario: pd.DataFrame,
    robust_realization: pd.DataFrame,
    robust_scenario: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []

    def _append(
        check_type: str,
        dataset_tag: str,
        scope_key: str,
        frequency_hz: str,
        lambda_per_m2: str,
        region_R_m: str,
        metric_name: str,
        metric_value: float,
        threshold: str = "",
        pass_flag: str = "",
    ) -> None:
        f_txt = "na" if frequency_hz == "" else f"f={frequency_hz}"
        l_txt = "na" if lambda_per_m2 == "" else f"lambda={lambda_per_m2}"
        r_txt = "na" if region_R_m == "" else f"R={region_R_m}"
        threshold_txt = "na" if threshold == "" else str(threshold)
        pass_txt = "na" if pass_flag == "" else str(pass_flag)
        rows.append(
            {
                "check_type": check_type,
                "dataset_tag": dataset_tag,
                "scope_key": scope_key,
                "frequency_hz": f_txt,
                "lambda_per_m2": l_txt,
                "region_R_m": r_txt,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "threshold": threshold_txt,
                "pass_flag": pass_txt,
            }
        )

    # Poisson diagnostics by setting, pooled (lambda, R), and aggregate.
    for tag, df in [
        ("baseline_fullband_2kHz", baseline_realization),
        ("robust_subsetfreq", robust_realization),
    ]:
        for sid, grp in df.groupby("setting_id", sort=True):
            d = metrics.poisson_diagnostics_from_counts(grp["N"].to_numpy(dtype=float))
            f_hz = f"{float(grp['frequency_hz'].iloc[0]):g}"
            lmb = f"{float(grp['lambda_per_m2'].iloc[0]):g}"
            rad = f"{float(grp['region_R_m'].iloc[0]):g}"
            key = f"setting_id={int(sid)}"
            _append("poisson_by_setting", tag, key, f_hz, lmb, rad, "mean_n", d["mean_n"])
            _append("poisson_by_setting", tag, key, f_hz, lmb, rad, "var_n", d["var_n"])
            _append("poisson_by_setting", tag, key, f_hz, lmb, rad, "fano", d["fano"])
        for (lmb, rad), grp in df.groupby(["lambda_per_m2", "region_R_m"], sort=True):
            d = metrics.poisson_diagnostics_from_counts(grp["N"].to_numpy(dtype=float))
            key = f"lambda={lmb:g}|R={rad:g}"
            _append(
                "poisson_pooled_lambda_R",
                tag,
                key,
                "",
                f"{float(lmb):g}",
                f"{float(rad):g}",
                "mean_n",
                d["mean_n"],
            )
            _append(
                "poisson_pooled_lambda_R",
                tag,
                key,
                "",
                f"{float(lmb):g}",
                f"{float(rad):g}",
                "var_n",
                d["var_n"],
            )
            _append(
                "poisson_pooled_lambda_R",
                tag,
                key,
                "",
                f"{float(lmb):g}",
                f"{float(rad):g}",
                "fano",
                d["fano"],
            )
        d_all = metrics.poisson_diagnostics_from_counts(df["N"].to_numpy(dtype=float))
        _append("poisson_aggregate_overall", tag, "all", "", "", "", "mean_n", d_all["mean_n"])
        _append("poisson_aggregate_overall", tag, "all", "", "", "", "var_n", d_all["var_n"])
        _append("poisson_aggregate_overall", tag, "all", "", "", "", "fano", d_all["fano"])

    # Cancellation diagnostics (uniform-phase scenarios).
    for sid, grp in baseline_realization.groupby("setting_id", sort=True):
        c = metrics.cancellation_diagnostics(grp)
        f_hz = f"{float(grp['frequency_hz'].iloc[0]):g}"
        lmb = f"{float(grp['lambda_per_m2'].iloc[0]):g}"
        rad = f"{float(grp['region_R_m'].iloc[0]):g}"
        key = f"setting_id={int(sid)}"
        _append(
            "cancellation_by_setting",
            "baseline_fullband_2kHz",
            key,
            f_hz,
            lmb,
            rad,
            "mean_re",
            c["mean_re"],
        )
        _append(
            "cancellation_by_setting",
            "baseline_fullband_2kHz",
            key,
            f_hz,
            lmb,
            rad,
            "mean_im",
            c["mean_im"],
        )
        _append(
            "cancellation_by_setting",
            "baseline_fullband_2kHz",
            key,
            f_hz,
            lmb,
            rad,
            "mean_abs",
            c["mean_abs"],
        )
    c_all = metrics.cancellation_diagnostics(baseline_realization)
    _append(
        "cancellation_aggregate",
        "baseline_fullband_2kHz",
        "all",
        "",
        "",
        "",
        "mean_re",
        c_all["mean_re"],
    )
    _append(
        "cancellation_aggregate",
        "baseline_fullband_2kHz",
        "all",
        "",
        "",
        "",
        "mean_im",
        c_all["mean_im"],
    )
    _append(
        "cancellation_aggregate",
        "baseline_fullband_2kHz",
        "all",
        "",
        "",
        "",
        "mean_abs",
        c_all["mean_abs"],
    )

    robust_uniform = robust_realization[robust_realization["phase_model"] == "uniform"]
    for sid, grp in robust_uniform.groupby("setting_id", sort=True):
        c = metrics.cancellation_diagnostics(grp)
        f_hz = f"{float(grp['frequency_hz'].iloc[0]):g}"
        lmb = f"{float(grp['lambda_per_m2'].iloc[0]):g}"
        rad = f"{float(grp['region_R_m'].iloc[0]):g}"
        key = f"setting_id={int(sid)}"
        _append(
            "cancellation_by_setting",
            "robust_subsetfreq_uniform",
            key,
            f_hz,
            lmb,
            rad,
            "mean_re",
            c["mean_re"],
        )
        _append(
            "cancellation_by_setting",
            "robust_subsetfreq_uniform",
            key,
            f_hz,
            lmb,
            rad,
            "mean_im",
            c["mean_im"],
        )
        _append(
            "cancellation_by_setting",
            "robust_subsetfreq_uniform",
            key,
            f_hz,
            lmb,
            rad,
            "mean_abs",
            c["mean_abs"],
        )
    if not robust_uniform.empty:
        c_all = metrics.cancellation_diagnostics(robust_uniform)
        _append(
            "cancellation_aggregate",
            "robust_subsetfreq_uniform",
            "all",
            "",
            "",
            "",
            "mean_re",
            c_all["mean_re"],
        )
        _append(
            "cancellation_aggregate",
            "robust_subsetfreq_uniform",
            "all",
            "",
            "",
            "",
            "mean_im",
            c_all["mean_im"],
        )
        _append(
            "cancellation_aggregate",
            "robust_subsetfreq_uniform",
            "all",
            "",
            "",
            "",
            "mean_abs",
            c_all["mean_abs"],
        )

    # Denominator diagnostics by setting and aggregate for both datasets.
    for tag, scen in [
        ("baseline_fullband_2kHz", baseline_scenario),
        ("robust_subsetfreq", robust_scenario),
    ]:
        vals = scen["denom_mag_min_over_domain"].to_numpy(dtype=float)
        d = metrics.denom_diagnostics(vals)
        _append(
            "denominator_aggregate",
            tag,
            "all",
            "",
            "",
            "",
            "denom_min",
            d["min"],
            threshold=f"{config.GATE_DENOM_MIN:g}",
            pass_flag=str(bool(d["min"] >= config.GATE_DENOM_MIN)),
        )
        _append("denominator_aggregate", tag, "all", "", "", "", "denom_p01", d["p01"])
        _append("denominator_aggregate", tag, "all", "", "", "", "denom_p05", d["p05"])
        _append("denominator_aggregate", tag, "all", "", "", "", "denom_median", d["median"])
        for _, row in scen.iterrows():
            denom = float(row["denom_mag_min_over_domain"])
            _append(
                "denominator_by_setting",
                tag,
                f"setting_id={int(row['setting_id'])}",
                f"{float(row['frequency_hz']):g}",
                f"{float(row['lambda_per_m2']):g}",
                f"{float(row['region_R_m']):g}",
                "denom_min",
                denom,
                threshold=f"{config.GATE_DENOM_MIN:g}",
                pass_flag=str(bool(denom >= config.GATE_DENOM_MIN)),
            )

    df = pd.DataFrame(rows)
    path = out_dir / "tableS1_sanity_checks.csv"
    return path, _write_table(df, path)


def tableS2_uncertainty_method(out_dir: Path) -> tuple[Path, int]:
    rows = [
        {
            "method_id": "U1",
            "metric": "mean_vmag",
            "ci_method": "bootstrap_percentile",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "ci_level": config.CI_LEVEL,
            "rng_seed": config.RNG_SEED,
            "notes": "Applied on realization magnitudes per frequency setting.",
        },
        {
            "method_id": "U2",
            "metric": "rms_vmag",
            "ci_method": "bootstrap_percentile",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "ci_level": config.CI_LEVEL,
            "rng_seed": config.RNG_SEED,
            "notes": "Applied on realization magnitudes per frequency setting.",
        },
        {
            "method_id": "U3",
            "metric": "p95_vmag,p99_vmag",
            "ci_method": "bootstrap_percentile",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "ci_level": config.CI_LEVEL,
            "rng_seed": config.RNG_SEED,
            "notes": "Percentile CI from bootstrap empirical distribution.",
        },
        {
            "method_id": "U4",
            "metric": "exceedance_tau",
            "ci_method": "bootstrap_percentile",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "ci_level": config.CI_LEVEL,
            "rng_seed": config.RNG_SEED,
            "notes": "Binary indicator bootstrap for P(|V|>tau).",
        },
    ]
    df = pd.DataFrame(rows)
    path = out_dir / "tableS2_uncertainty_method.csv"
    return path, _write_table(df, path)


def tableS_validation_gates(
    baseline_realization: pd.DataFrame,
    baseline_scenario: pd.DataFrame,
    robust_realization: pd.DataFrame,
    robust_scenario: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []

    # gate_poisson on homogeneous pools; overall uses weighted pooled deviation.
    poisson_group_rows: list[dict[str, Any]] = []
    for (lmb, rad), grp in baseline_realization.groupby(["lambda_per_m2", "region_R_m"], sort=True):
        d = metrics.poisson_diagnostics_from_counts(grp["N"].to_numpy(dtype=float))
        poisson_group_rows.append(
            {
                "gate_name": "gate_poisson",
                "scope": "pooled_lambda_R",
                "interpretation": "acceptance",
                "dataset_tag": "baseline_fullband_2kHz",
                "group_key": f"lambda={lmb:g}|R={rad:g}",
                "n_samples": int(grp.shape[0]),
                "metric_1_name": "fano",
                "metric_1_value": d["fano"],
                "metric_2_name": "abs(fano-1)",
                "metric_2_value": abs(d["fano"] - 1.0),
                "threshold": config.GATE_POISSON_TOL,
                "pass_flag": bool(abs(d["fano"] - 1.0) <= config.GATE_POISSON_TOL),
            }
        )
    rows.extend(poisson_group_rows)
    poisson_df = pd.DataFrame(poisson_group_rows)
    weights = poisson_df["n_samples"].to_numpy(dtype=float)
    weighted_abs_dev = float(
        np.average(poisson_df["metric_2_value"].to_numpy(dtype=float), weights=weights)
    )
    weighted_fano = float(
        np.average(poisson_df["metric_1_value"].to_numpy(dtype=float), weights=weights)
    )
    rows.append(
        {
            "gate_name": "gate_poisson",
            "scope": "overall",
            "interpretation": "acceptance",
            "dataset_tag": "baseline_fullband_2kHz",
            "group_key": "weighted_over_pooled_lambda_R",
            "n_samples": int(poisson_df["n_samples"].sum()),
            "metric_1_name": "weighted_fano",
            "metric_1_value": weighted_fano,
            "metric_2_name": "weighted_abs(fano-1)",
            "metric_2_value": weighted_abs_dev,
            "threshold": config.GATE_POISSON_TOL,
            "pass_flag": bool(weighted_abs_dev <= config.GATE_POISSON_TOL),
        }
    )

    # gate_cancellation for uniform-phase scenarios on pooled groups.
    def _cancellation_rows(
        df: pd.DataFrame,
        tag: str,
        group_cols: list[str],
    ) -> list[dict[str, Any]]:
        out_local: list[dict[str, Any]] = []
        for key, grp in df.groupby(group_cols, sort=True):
            c = metrics.cancellation_diagnostics(grp)
            lim = config.GATE_CANCELLATION_FACTOR * c["mean_abs"]
            passed = (abs(c["mean_re"]) <= lim) and (abs(c["mean_im"]) <= lim)

            def _fmt(v: Any) -> str:
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return f"{float(v):g}"
                return str(v)

            if isinstance(key, tuple):
                key_txt = "|".join(f"{col}={_fmt(val)}" for col, val in zip(group_cols, key))
            else:
                key_txt = f"{group_cols[0]}={_fmt(key)}"
            out_local.append(
                {
                    "gate_name": "gate_cancellation",
                    "scope": "pooled",
                    "interpretation": "diagnostic",
                    "dataset_tag": tag,
                    "group_key": key_txt,
                    "n_samples": int(grp.shape[0]),
                    "metric_1_name": "|mean(Re(V))|",
                    "metric_1_value": abs(c["mean_re"]),
                    "metric_2_name": "|mean(Im(V))|",
                    "metric_2_value": abs(c["mean_im"]),
                    "threshold": lim,
                    "pass_flag": bool(passed),
                }
            )
        c_all = metrics.cancellation_diagnostics(df)
        lim_all = config.GATE_CANCELLATION_FACTOR * c_all["mean_abs"]
        pass_all = (abs(c_all["mean_re"]) <= lim_all) and (abs(c_all["mean_im"]) <= lim_all)
        out_local.append(
            {
                "gate_name": "gate_cancellation",
                "scope": "overall_uniform",
                "interpretation": "acceptance",
                "dataset_tag": tag,
                "group_key": "all_uniform_phase",
                "n_samples": int(df.shape[0]),
                "metric_1_name": "|mean(Re(V))|",
                "metric_1_value": abs(c_all["mean_re"]),
                "metric_2_name": "|mean(Im(V))|",
                "metric_2_value": abs(c_all["mean_im"]),
                "threshold": lim_all,
                "pass_flag": bool(pass_all),
            }
        )
        return out_local

    rows.extend(
        _cancellation_rows(
            baseline_realization,
            "baseline_fullband_2kHz",
            ["lambda_per_m2", "region_R_m"],
        )
    )
    robust_uniform = robust_realization[robust_realization["phase_model"] == "uniform"]
    if not robust_uniform.empty:
        rows.extend(
            _cancellation_rows(
                robust_uniform,
                "robust_subsetfreq_uniform",
                ["scenario_name", "lambda_per_m2", "region_R_m"],
            )
        )

    # gate_denom from scenario matrices
    for tag, scen in [
        ("baseline_fullband_2kHz", baseline_scenario),
        ("robust_subsetfreq", robust_scenario),
    ]:
        for _, r in scen.iterrows():
            denom = float(r["denom_mag_min_over_domain"])
            rows.append(
                {
                    "gate_name": "gate_denom",
                    "scope": "per_setting",
                    "interpretation": "acceptance",
                    "dataset_tag": tag,
                    "group_key": str(int(r["setting_id"])),
                    "n_samples": 1,
                    "metric_1_name": "denom_mag_min_over_domain",
                    "metric_1_value": denom,
                    "metric_2_name": "margin_to_threshold",
                    "metric_2_value": denom - config.GATE_DENOM_MIN,
                    "threshold": config.GATE_DENOM_MIN,
                    "pass_flag": bool(denom >= config.GATE_DENOM_MIN),
                }
            )

    detail_df = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for gate, gdf in detail_df.groupby("gate_name"):
        eval_df = gdf
        if gate == "gate_cancellation":
            eval_df = gdf[gdf["scope"] == "overall_uniform"].copy()
            if eval_df.empty:
                eval_df = gdf
        n_pass = int(eval_df["pass_flag"].sum())
        n_total = int(eval_df.shape[0])
        n_fail = n_total - n_pass
        if gate == "gate_poisson":
            violations = eval_df["metric_2_value"] - eval_df["threshold"]
        elif gate == "gate_cancellation":
            violations = (
                np.maximum(eval_df["metric_1_value"], eval_df["metric_2_value"])
                - eval_df["threshold"]
            )
        else:
            violations = eval_df["threshold"] - eval_df["metric_1_value"]
        worst_idx = int(np.argmax(violations.to_numpy(dtype=float)))
        worst = eval_df.iloc[worst_idx]
        summary_rows.append(
            {
                "gate_name": gate,
                "scope": "summary",
                "interpretation": "summary_acceptance",
                "dataset_tag": "all",
                "group_key": str(worst["group_key"]),
                "n_samples": n_total,
                "metric_1_name": "pass_count",
                "metric_1_value": n_pass,
                "metric_2_name": "fail_count",
                "metric_2_value": n_fail,
                "threshold": worst["threshold"],
                "pass_flag": bool(n_fail == 0),
            }
        )

    df = pd.concat([detail_df, pd.DataFrame(summary_rows)], ignore_index=True)
    path = out_dir / "tableS_validation_gates.csv"
    return path, _write_table(df, path)


def tableS7_cancellation_diagnostics(
    baseline_realization: pd.DataFrame,
    robust_realization: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []

    def _fmt_key(key: Any, cols: list[str]) -> str:
        if isinstance(key, tuple):
            vals = key
        else:
            vals = (key,)
        parts = []
        for c, v in zip(cols, vals):
            if isinstance(v, (int, float, np.integer, np.floating)):
                parts.append(f"{c}={float(v):g}")
            else:
                parts.append(f"{c}={str(v)}")
        return "|".join(parts)

    def _append_group_rows(df: pd.DataFrame, dataset_tag: str, group_cols: list[str]) -> None:
        if df.empty:
            return
        for key, grp in df.groupby(group_cols, sort=True):
            c = metrics.cancellation_diagnostics(grp)
            abs_re = abs(c["mean_re"])
            abs_im = abs(c["mean_im"])
            limit = config.GATE_CANCELLATION_FACTOR * c["mean_abs"]
            max_comp = max(abs_re, abs_im)
            pass_flag = bool(max_comp <= limit)
            rows.append(
                {
                    "dataset_tag": dataset_tag,
                    "scope": "pooled",
                    "group_key": _fmt_key(key, group_cols),
                    "n_samples": int(grp.shape[0]),
                    "abs_mean_re": abs_re,
                    "abs_mean_im": abs_im,
                    "mean_abs": c["mean_abs"],
                    "threshold": limit,
                    "max_component": max_comp,
                    "violation": max(0.0, max_comp - limit),
                    "violation_ratio": max_comp / max(limit, 1e-12),
                    "pass_flag": pass_flag,
                    "interpretation": "diagnostic_only",
                }
            )

        c_all = metrics.cancellation_diagnostics(df)
        abs_re = abs(c_all["mean_re"])
        abs_im = abs(c_all["mean_im"])
        limit = config.GATE_CANCELLATION_FACTOR * c_all["mean_abs"]
        max_comp = max(abs_re, abs_im)
        rows.append(
            {
                "dataset_tag": dataset_tag,
                "scope": "overall_uniform",
                "group_key": "all_uniform_phase",
                "n_samples": int(df.shape[0]),
                "abs_mean_re": abs_re,
                "abs_mean_im": abs_im,
                "mean_abs": c_all["mean_abs"],
                "threshold": limit,
                "max_component": max_comp,
                "violation": max(0.0, max_comp - limit),
                "violation_ratio": max_comp / max(limit, 1e-12),
                "pass_flag": bool(max_comp <= limit),
                "interpretation": "acceptance_context",
            }
        )

    _append_group_rows(
        baseline_realization,
        "baseline_fullband_2kHz",
        ["lambda_per_m2", "region_R_m"],
    )
    robust_uniform = robust_realization[robust_realization["phase_model"] == "uniform"].copy()
    _append_group_rows(
        robust_uniform,
        "robust_subsetfreq_uniform",
        ["scenario_name", "lambda_per_m2", "region_R_m"],
    )

    out = pd.DataFrame(rows).sort_values(
        ["dataset_tag", "scope", "pass_flag", "violation_ratio"],
        ascending=[True, True, True, False],
    )
    path = out_dir / "tableS7_cancellation_diagnostics.csv"
    return path, _write_table(out, path)


def tableS_missing_assets(
    outline_text: str,
    paths: config.Paths,
    out_dir: Path,
) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []
    txt = outline_text.lower()
    # Outline requests feeder benchmark assets (Table 5/6, Figure 7).
    needs_feeder = any(
        re.search(pattern, txt) is not None
        for pattern in [
            r"table\s*5",
            r"table\s*6",
            r"figure\s*7",
            r"fig\.?\s*7",
        ]
    )
    feeder_assets_ready = (
        paths.feeder_benchmark_kernels.exists()
        and paths.feeder_benchmark_features.exists()
        and paths.feeder_model_realizations.exists()
        and paths.feeder_model_setting_summary.exists()
    )
    if needs_feeder and not feeder_assets_ready:
        rows.extend(
            [
                {
                    "requested_item": "Table 5 (feeder benchmark specification)",
                    "status": "missing_input_assets",
                    "reason": "No feeder topology/cable/transformer benchmark dataset provided.",
                    "required_inputs": "feeder topology class, cable parameters, transformer impedance, fitted kernel parameters",
                    "available_synthetic_inputs": "PPP and robustness synthetic aggregate realizations only",
                },
                {
                    "requested_item": "Table 6 (analytical vs feeder-simulated metrics)",
                    "status": "missing_input_assets",
                    "reason": "No feeder-simulated outputs available for comparison.",
                    "required_inputs": "paired analytical and feeder simulation outputs at matched settings",
                    "available_synthetic_inputs": "analytical-style synthetic summaries only",
                },
                {
                    "requested_item": "Figure 7 (predicted vs feeder simulated scatter)",
                    "status": "missing_input_assets",
                    "reason": "No feeder simulation points provided.",
                    "required_inputs": "feeder simulated RMS/high-percentile points",
                    "available_synthetic_inputs": "baseline/robust synthetic Monte Carlo outputs",
                },
            ]
        )
    elif needs_feeder and feeder_assets_ready:
        rows.append(
            {
                "requested_item": "Table 5/6 and Figure 7 feeder benchmark assets",
                "status": "available",
                "reason": "feeder-model benchmark assets generated in this run",
                "required_inputs": "kernel + feeder-model realizations + feeder summary",
                "available_synthetic_inputs": "feeder_benchmark_kernels.csv; feeder_model_simulated_realizations.csv; feeder_model_setting_summary.csv",
            }
        )
    df = pd.DataFrame(rows)
    path = out_dir / "tableS_missing_assets.csv"
    return path, _write_table(df, path)


def tableS3_synthetic_data_generator_subsection(
    baseline_scenario: pd.DataFrame,
    robust_scenario: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, int]:
    b_freq = np.sort(baseline_scenario["frequency_hz"].unique())
    b_lambda = np.sort(baseline_scenario["lambda_per_m2"].unique())
    b_r = np.sort(baseline_scenario["region_R_m"].unique())
    k_vals = np.sort(robust_scenario["kappa"].dropna().unique())
    phase_models = ", ".join(sorted(robust_scenario["phase_model"].dropna().astype(str).unique()))
    current_models = ", ".join(
        sorted(robust_scenario["current_model"].dropna().astype(str).unique())
    )
    y_models = ", ".join(sorted(robust_scenario["Y_model"].dropna().astype(str).unique()))
    scen_list = ", ".join(sorted(robust_scenario["scenario_name"].dropna().astype(str).unique()))

    text = "\n".join(
        [
            "# Synthetic data generator subsection",
            "",
            "## Sampling workflow (pseudocode)",
            "- Select setting `(f, lambda, R, scenario_name, tau, seed_setting)` from scenario matrix.",
            "- Compute study area `A = pi R^2` and nominal mean source count `mu = lambda * A`.",
            "- Draw source count `N ~ Poisson(mu)` for baseline PPP settings.",
            "- Draw marks `(phase, current, Y)` using scenario-specific mark model.",
            "- Evaluate per-source contribution and aggregate `V = sum_i contribution_i`.",
            "- Store realizations `{N, Re(V), Im(V), |V|, denom_mag_min_over_domain}` and setting summaries.",
            "",
            "## Distributions and parameters used (from available synthetic assets)",
            f"- Baseline frequency grid: {b_freq.min():.0f}-{b_freq.max():.0f} Hz with `Delta f={float(baseline_scenario['band_width_hz'].median()):.0f}` Hz.",
            f"- Baseline lambda levels [1/m^2]: {', '.join(f'{v:g}' for v in b_lambda)}.",
            f"- Baseline radius levels [m]: {', '.join(f'{v:g}' for v in b_r)}.",
            "- Baseline phase model: `uniform` (uniform angle on [-pi, pi]).",
            f"- Robustness scenarios available: {scen_list}.",
            f"- Robustness phase models: {phase_models}.",
            f"- Robustness coherence parameter(s) (von Mises kappa): {', '.join(f'{v:g}' for v in k_vals) if len(k_vals) else 'not recorded'}.",
            f"- Robustness current models: {current_models}.",
            f"- Robustness Y models: {y_models}.",
            "- Clustered spatial model present as `clustered_thomas`; cluster hyperparameters are not explicitly encoded in the CSV schema.",
            "",
            "> Parameters are not calibrated; used only to validate derivations.",
            "",
            "This subsection is auto-generated from the provided synthetic scenario matrices and does not fabricate unavailable feeder assets.",
        ]
    )
    path = out_dir / "tableS3_synthetic_data_generator_subsection.md"
    return path, _write_text(text, path)


def generate_all_tables(inputs: dict[str, Any], paths: config.Paths) -> list[tuple[Path, int]]:
    outputs: list[tuple[Path, int]] = []
    outputs.append(
        table1_scenario_matrix(
            inputs["baseline_scenario"], inputs["robust_scenario"], paths.tables_dir
        )
    )
    outputs.append(table2_analytical_summary(inputs["concept_text"], paths.tables_dir))
    outputs.append(
        table3_baseline_results_by_frequency(
            inputs["baseline_realization"], inputs["baseline_scenario"], paths.tables_dir
        )
    )
    outputs.append(table4_robustness_effect_sizes(inputs["robust_setting"], paths.tables_dir))
    outputs.append(table5_feeder_model_kernel_features(paths, paths.tables_dir))
    outputs.append(table6_analytical_vs_feeder_model_metrics(paths, paths.tables_dir))
    outputs.append(tableS6_feeder_model_comparison_summary(paths, paths.tables_dir))
    outputs.append(
        tableS1_sanity_checks(
            inputs["baseline_realization"],
            inputs["baseline_scenario"],
            inputs["robust_realization"],
            inputs["robust_scenario"],
            paths.tables_dir,
        )
    )
    outputs.append(tableS2_uncertainty_method(paths.tables_dir))
    outputs.append(
        tableS_validation_gates(
            inputs["baseline_realization"],
            inputs["baseline_scenario"],
            inputs["robust_realization"],
            inputs["robust_scenario"],
            paths.tables_dir,
        )
    )
    outputs.append(
        tableS7_cancellation_diagnostics(
            inputs["baseline_realization"],
            inputs["robust_realization"],
            paths.tables_dir,
        )
    )
    outputs.append(
        tableS3_synthetic_data_generator_subsection(
            inputs["baseline_scenario"],
            inputs["robust_scenario"],
            paths.tables_dir,
        )
    )
    outputs.append(tableS_missing_assets(inputs["outline_text"], paths, paths.tables_dir))
    return outputs
