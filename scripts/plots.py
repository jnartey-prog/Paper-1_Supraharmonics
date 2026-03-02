"""Figure generation for manuscript outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from scripts import config, io

try:
    from scripts.plotting_style import apply_manuscript_style, save_figure_bundle
except ModuleNotFoundError:  # pragma: no cover
    from plotting_style import apply_manuscript_style, save_figure_bundle


apply_manuscript_style()


def _save(fig: plt.Figure, path: Path) -> None:
    save_figure_bundle(fig, path, dpi=config.PLOT_DPI)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.tick_params(labelsize=9)


def _scenario_label(name: str) -> str:
    mapping = {
        "baseline_ppp": "Baseline PPP",
        "clustered_thomas": "Clustered (Thomas)",
        "partial_coherence_kappa_10": r"Partial coherence ($\kappa=10$)",
        "inhomog_hotspot": "Inhomogeneous hotspot",
    }
    return mapping.get(name, name)


def _add_caption_line(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8, color="#444444")


def _sci_lmbda(x: float, _: float) -> str:
    if x <= 0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    coeff = x / (10**exp)
    if abs(coeff - 1.0) < 1e-12:
        return rf"$10^{{{exp}}}$"
    return rf"${coeff:.1f}\times 10^{{{exp}}}$"


def _read_required_csv(path: Path, required_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df


def _nearest_freq(df: pd.DataFrame, target_hz: float) -> float:
    vals = sorted(df["frequency_hz"].unique().tolist())
    return io.nearest_value(vals, target_hz)


def _empirical_ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    n = len(x)
    y = 1.0 - (np.arange(1, n + 1, dtype=float) / n) + (1.0 / n)
    return x, y


def fig1_kernel_magnitude_vs_distance(paths: config.Paths) -> Path:
    k = _read_required_csv(
        paths.feeder_benchmark_kernels,
        ["feeder_id", "feeder_name", "frequency_hz", "distance_m", "Ztr_mag_ohm"],
    )
    target_freqs = np.array([10_000.0, 30_000.0, 80_000.0, 150_000.0], dtype=float)
    available = np.sort(k["frequency_hz"].unique())
    selected = [io.nearest_value(available.tolist(), float(f)) for f in target_freqs]

    feeders = sorted(k["feeder_id"].unique().tolist())
    fig, axes = plt.subplots(1, len(feeders), figsize=(14.0, 4.4), sharey=True)
    if len(feeders) == 1:
        axes = [axes]

    cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, feeder_id in zip(axes, feeders):
        sub = k[k["feeder_id"] == feeder_id].copy()
        feeder_name = str(sub["feeder_name"].iloc[0])
        for color, freq in zip(cmap, selected):
            sf = sub[np.isclose(sub["frequency_hz"], freq)].sort_values("distance_m")
            ax.plot(
                sf["distance_m"].to_numpy(dtype=float),
                sf["Ztr_mag_ohm"].to_numpy(dtype=float),
                lw=2.0,
                color=color,
                label=f"{freq / 1000:.0f} kHz",
            )
        ax.set_title(f"Feeder {feeder_id}: {feeder_name}", fontsize=10)
        ax.set_xlabel(r"Electrical distance, $d$ [m]")
        _style_axes(ax)
    axes[0].set_ylabel(r"Transfer-impedance magnitude, $|Z_{\mathrm{tr}}(f,d)|$ [$\Omega$]")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Figure 1. Transfer-impedance kernel magnitude versus distance",
        fontsize=12,
        fontweight="semibold",
    )
    _add_caption_line(
        fig,
        "Feeder-model benchmark; frequency grid: 2-150 kHz (Δf = 2 kHz); distance grid: 0-1000 m; datasets: feeder kernels.",
    )
    out = paths.figures_dir / "fig1_kernel_magnitude_vs_distance.png"
    _save(fig, out)
    return out


def fig2_rms_vs_lambda(inputs: dict[str, Any], paths: config.Paths) -> Path:
    setting = inputs["baseline_setting"]
    scenario = inputs["baseline_scenario"]
    r_val = io.nearest_value(
        sorted(setting["region_R_m"].unique().tolist()), config.BASELINE_TARGET_R
    )
    targets_hz = [k * 1000.0 for k in config.BASELINE_TARGET_FREQ_FIG2_KHZ]
    selected = [_nearest_freq(setting, t) for t in targets_hz]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for freq in selected:
        sub = setting[
            (np.isclose(setting["region_R_m"], r_val)) & (np.isclose(setting["frequency_hz"], freq))
        ].sort_values("lambda_per_m2")
        ax.loglog(
            sub["lambda_per_m2"].to_numpy(),
            sub["rms_Vmag"].to_numpy(),
            marker="o",
            lw=2.0,
            label=f"{freq / 1000:.0f} kHz",
        )
    ax.set_xlabel(r"Active charger density, $\lambda$ [m$^{-2}$]")
    ax.set_ylabel(r"$V_{\mathrm{RMS}}$ [V]")
    ax.set_title(
        r"Figure 2. RMS aggregate voltage versus charger density ($R=500$ m)",
        fontsize=12,
        fontweight="semibold",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_sci_lmbda))
    _style_axes(ax)
    ax.legend(frameon=False)
    _add_caption_line(
        fig,
        f"Baseline PPP; K={int(scenario['n_realizations'].median())}; frequency grid: 2-150 kHz (Δf = {int(scenario['band_width_hz'].median())} Hz).",
    )
    out = paths.figures_dir / "fig2_rms_vs_lambda.png"
    _save(fig, out)
    return out


def fig3_p99_contour_lambda_R(inputs: dict[str, Any], paths: config.Paths) -> Path:
    setting = inputs["baseline_setting"]
    scenario = inputs["baseline_scenario"]
    f_ref = _nearest_freq(setting, config.BASELINE_TARGET_FREQ_FIG3_HZ)
    sub = setting[np.isclose(setting["frequency_hz"], f_ref)].copy()
    pivot = sub.pivot(index="region_R_m", columns="lambda_per_m2", values="p99_Vmag").sort_index()
    lambdas = pivot.columns.to_numpy(dtype=float)
    radii = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_yticks(np.arange(len(radii)))
    ax.set_xticklabels([_sci_lmbda(float(v), 0.0) for v in lambdas], rotation=35, ha="right")
    ax.set_yticklabels([f"{v:g}" for v in radii])
    for i in range(len(radii)):
        for j in range(len(lambdas)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=r"$P_{99}(|V|)$ [V]")
    ax.set_xlabel(r"Active charger density, $\lambda$ [m$^{-2}$]")
    ax.set_ylabel(r"Service radius, $R$ [m]")
    ax.set_title(
        rf"Figure 3. $P_{{99}}(|V|)$ over $(\lambda, R)$ at $f={f_ref / 1000:.0f}$ kHz",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax)
    _add_caption_line(
        fig,
        f"Baseline PPP heatmap; K={int(scenario['n_realizations'].median())}; frequency grid: 2-150 kHz (Δf = {int(scenario['band_width_hz'].median())} Hz).",
    )
    out = paths.figures_dir / "fig3_p99_contour_lambda_R.png"
    _save(fig, out)
    return out


def fig4_ccdf_ppp_vs_clustered(inputs: dict[str, Any], paths: config.Paths) -> Path:
    r = inputs["robust_realization"]
    rs = inputs["robust_scenario"]
    f_ref = _nearest_freq(r, config.ROBUST_TARGET_FREQ_HZ)
    sub = r[
        (np.isclose(r["frequency_hz"], f_ref))
        & (np.isclose(r["lambda_per_m2"], config.BASELINE_TARGET_LAMBDA))
        & (np.isclose(r["region_R_m"], config.BASELINE_TARGET_R))
        & (
            r["scenario_name"].isin(
                [
                    config.ROBUST_TARGET_SCENARIOS["baseline"],
                    config.ROBUST_TARGET_SCENARIOS["clustered"],
                ]
            )
        )
    ].copy()
    if sub.empty:
        raise ValueError("No robust realization rows found for fig4 target selection.")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for scen, color in [
        (config.ROBUST_TARGET_SCENARIOS["baseline"], "#1f77b4"),
        (config.ROBUST_TARGET_SCENARIOS["clustered"], "#d62728"),
    ]:
        vals = sub[sub["scenario_name"] == scen]["Vagg_mag_V"].to_numpy(dtype=float)
        if vals.size == 0:
            raise ValueError(
                f"No robust realization rows found for scenario '{scen}' in fig4 selection."
            )
        x, y = _empirical_ccdf(vals)
        ax.semilogy(x, y, lw=2.0, color=color, label=_scenario_label(scen))
    ax.set_xlabel(r"Voltage magnitude threshold, $v$ [V]")
    ax.set_ylabel(r"CCDF, $\Pr(|V| > v)$")
    ax.set_title(
        rf"Figure 4. CCDF comparison: baseline PPP vs clustered model at $f={f_ref / 1000:.0f}$ kHz",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax)
    ax.legend(frameon=False)
    _add_caption_line(
        fig,
        f"Feeder-model robustness subset; K={int(rs['n_realizations'].median())}; Δf = {int(rs['band_width_hz'].median())} Hz; λ={config.BASELINE_TARGET_LAMBDA:g} m^-2, R={int(config.BASELINE_TARGET_R)} m.",
    )
    out = paths.figures_dir / "fig4_ccdf_ppp_vs_clustered.png"
    _save(fig, out)
    return out


def fig5_phase_coherence_sweep(inputs: dict[str, Any], paths: config.Paths) -> Path:
    s = inputs["robust_setting"]
    rs = inputs["robust_scenario"]
    f_ref = _nearest_freq(s, config.ROBUST_TARGET_FREQ_HZ)
    sub = s[
        (np.isclose(s["frequency_hz"], f_ref))
        & (np.isclose(s["lambda_per_m2"], config.BASELINE_TARGET_LAMBDA))
        & (np.isclose(s["region_R_m"], config.BASELINE_TARGET_R))
        & (
            s["scenario_name"].isin(
                [
                    config.ROBUST_TARGET_SCENARIOS["baseline"],
                    config.ROBUST_TARGET_SCENARIOS["coherence"],
                ]
            )
        )
    ].copy()
    if sub.empty:
        raise ValueError("No robust setting rows found for fig5 target selection.")
    sub = sub.sort_values("scenario_name")
    x_labels = [_scenario_label(scn) for scn in sub["scenario_name"].tolist()]
    x = np.arange(len(x_labels))
    p99_vals = sub["p99_Vmag"].to_numpy(dtype=float)
    ex_vals = sub["exceed_tau"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(x, p99_vals, "o-", color="#1f77b4", lw=2.0, label="P99(|V|)")
    l2 = ax2.plot(x, ex_vals, "s--", color="#d62728", lw=2.0, label="Exceedance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=18, ha="right")
    ax1.set_ylabel(r"$P_{99}(|V|)$ [V]", color="#1f77b4")
    ax2.set_ylabel(r"Exceedance probability, $\Pr(|V|>\tau)$", color="#d62728")
    ax1.set_title(
        rf"Figure 5. Phase-coherence effect on $P_{{99}}(|V|)$ and exceedance at $f={f_ref / 1000:.0f}$ kHz",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax1)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="upper left")
    _add_caption_line(
        fig,
        f"Robustness subset; K={int(rs['n_realizations'].median())}; Δf = {int(rs['band_width_hz'].median())} Hz; λ={config.BASELINE_TARGET_LAMBDA:g} m^-2, R={int(config.BASELINE_TARGET_R)} m.",
    )
    out = paths.figures_dir / "fig5_phase_coherence_sweep.png"
    _save(fig, out)
    return out


def fig6_inhomogeneous_same_mean_density(inputs: dict[str, Any], paths: config.Paths) -> Path:
    s = inputs["robust_setting"]
    rs = inputs["robust_scenario"]
    f_ref = _nearest_freq(s, config.ROBUST_TARGET_FREQ_HZ)
    sub = s[
        (np.isclose(s["frequency_hz"], f_ref))
        & (np.isclose(s["lambda_per_m2"], config.BASELINE_TARGET_LAMBDA))
        & (np.isclose(s["region_R_m"], config.BASELINE_TARGET_R))
        & (
            s["scenario_name"].isin(
                [
                    config.ROBUST_TARGET_SCENARIOS["baseline"],
                    config.ROBUST_TARGET_SCENARIOS["inhomogeneous"],
                ]
            )
        )
    ].copy()
    if sub.empty:
        raise ValueError("No robust setting rows found for fig6 target selection.")
    sub = sub.sort_values("scenario_name")
    labels = [_scenario_label(scn) for scn in sub["scenario_name"].tolist()]
    p99_vals = sub["p99_Vmag"].to_numpy(dtype=float)
    ex_vals = sub["exceed_tau"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True)

    axes[0].bar(x, p99_vals, color="#1f77b4")
    axes[0].set_ylabel(r"$P_{99}(|V|)$ [V]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_title(r"Panel A: $P_{99}(|V|)$", fontsize=10)
    _style_axes(axes[0])

    axes[1].bar(x, ex_vals, color="#d62728")
    axes[1].set_ylabel(r"Exceedance probability, $\Pr(|V|>\tau)$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_title(r"Panel B: Exceedance", fontsize=10)
    _style_axes(axes[1])

    fig.suptitle(
        rf"Figure 6. Inhomogeneous deployment effect at $f={f_ref / 1000:.0f}$ kHz",
        fontsize=12,
        fontweight="semibold",
    )
    _add_caption_line(
        fig,
        f"Robustness subset; K={int(rs['n_realizations'].median())}; Δf = {int(rs['band_width_hz'].median())} Hz; λ={config.BASELINE_TARGET_LAMBDA:g} m^-2, R={int(config.BASELINE_TARGET_R)} m.",
    )
    out = paths.figures_dir / "fig6_inhomogeneous_same_mean_density.png"
    _save(fig, out)
    return out


def fig7_predicted_vs_feeder_scatter(paths: config.Paths) -> Path:
    t6 = _read_required_csv(
        paths.tables_dir / "table6_analytical_vs_feeder_model_metrics.csv",
        [
            "feeder_id",
            "rms_pred",
            "rms_sim",
            "p99_pred",
            "p99_sim",
        ],
    )
    df = t6.copy()
    if df.empty:
        raise ValueError("table6 has no rows for figure 7.")

    feeders = sorted(df["feeder_id"].unique().tolist())
    color_map = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for feeder_id in feeders:
        sub = df[df["feeder_id"] == feeder_id]
        axes[0].scatter(
            sub["rms_pred"],
            sub["rms_sim"],
            s=14,
            alpha=0.60,
            c=color_map.get(feeder_id, "#7f7f7f"),
            label=f"Feeder {feeder_id}",
        )
        axes[1].scatter(
            sub["p99_pred"],
            sub["p99_sim"],
            s=14,
            alpha=0.60,
            c=color_map.get(feeder_id, "#7f7f7f"),
        )

    for ax, xcol, ycol, label in [
        (axes[0], "rms_pred", "rms_sim", r"$V_{\mathrm{RMS}}$"),
        (axes[1], "p99_pred", "p99_sim", r"$P_{99}(|V|)$"),
    ]:
        min_v = float(min(df[xcol].min(), df[ycol].min()))
        max_v = float(max(df[xcol].max(), df[ycol].max()))
        ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.3, label="1:1 line")
        ax.set_xlabel(f"Predicted {label} [V]")
        ax.set_ylabel(f"Simulated {label} [V]")
        _style_axes(ax)

    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Panel A: RMS", fontsize=10)
    axes[1].set_title("Panel B: 99th percentile", fontsize=10)
    fig.suptitle(
        "Figure 7. Predicted versus feeder-model simulated metrics",
        fontsize=12,
        fontweight="semibold",
    )
    _add_caption_line(
        fig,
        "Feeder-model benchmark; analytical surrogate: Poisson-Rayleigh mixture tail proxy; frequency grid: 2-150 kHz (Δf = 2 kHz).",
    )
    out = paths.figures_dir / "fig7_predicted_vs_feeder_scatter.png"
    _save(fig, out)
    return out


def fig8_allowable_lambda_vs_frequency(inputs: dict[str, Any], paths: config.Paths) -> Path:
    s = inputs["baseline_setting"]
    bs = inputs["baseline_scenario"]
    r_val = io.nearest_value(sorted(s["region_R_m"].unique().tolist()), config.BASELINE_TARGET_R)
    sub = s[np.isclose(s["region_R_m"], r_val)].copy()
    freqs = sorted(sub["frequency_hz"].unique().tolist())
    allowable = []
    for freq in freqs:
        sf = sub[np.isclose(sub["frequency_hz"], freq)].sort_values("lambda_per_m2")
        feasible = sf[sf["p99_Vmag"] <= config.ALLOWABLE_LAMBDA_P99_LIMIT_V][
            "lambda_per_m2"
        ].tolist()
        allowable.append(max(feasible) if feasible else np.nan)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(np.array(freqs) / 1000.0, allowable, marker="o", lw=2.0)
    ax.set_xlabel(r"Frequency, $f$ [kHz]")
    ax.set_ylabel(r"Maximum allowable density, $\lambda^\ast$ [m$^{-2}$]")
    ax.set_title(
        rf"Figure 8. Allowable density $\lambda^\ast(f)$ from the constraint $P_{{99}}(|V|)\leq {config.ALLOWABLE_LAMBDA_P99_LIMIT_V:.2f}$ V",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax)
    _add_caption_line(
        fig,
        f"Baseline PPP; K={int(bs['n_realizations'].median())}; frequency grid: 2-150 kHz (Δf = {int(bs['band_width_hz'].median())} Hz); R={int(config.BASELINE_TARGET_R)} m.",
    )
    out = paths.figures_dir / "fig8_allowable_lambda_vs_frequency.png"
    _save(fig, out)
    return out


def figS1_denom_hist(inputs: dict[str, Any], paths: config.Paths) -> Path:
    b = inputs["baseline_scenario"]
    vals = b["denom_mag_min_over_domain"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(vals, bins=24, edgecolor="white")
    ax.axvline(
        config.GATE_DENOM_MIN, color="red", linestyle="--", lw=1.5, label=r"Gate threshold ($c_0$)"
    )
    ax.set_xlabel(r"Minimum denominator separation, $\min_d |1+Y(f)Z_{\mathrm{tr}}(f,d)|$")
    ax.set_ylabel("Count")
    ax.set_title(
        "Figure S1. Distribution of denominator separation",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax)
    ax.legend(frameon=False)
    _add_caption_line(
        fig,
        f"Baseline PPP; K={int(b['n_realizations'].median())}; frequency grid: 2-150 kHz (Δf = {int(b['band_width_hz'].median())} Hz).",
    )
    out = paths.figures_dir / "figS1_denom_separation_hist.png"
    _save(fig, out)
    return out


def figS2_fano_distribution(inputs: dict[str, Any], paths: config.Paths) -> Path:
    s = inputs["baseline_setting"]
    bs = inputs["baseline_scenario"]
    fano = s["var_N"].to_numpy(dtype=float) / np.maximum(s["mean_N"].to_numpy(dtype=float), 1e-12)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(fano, bins=24, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.3, label="Poisson ideal")
    ax.axvline(
        1.0 - config.GATE_POISSON_TOL, color="red", linestyle=":", lw=1.2, label="gate bounds"
    )
    ax.axvline(1.0 + config.GATE_POISSON_TOL, color="red", linestyle=":", lw=1.2)
    ax.set_xlabel(r"Fano factor, $\mathrm{Var}(N)/\mathbb{E}[N]$")
    ax.set_ylabel("Count of settings")
    ax.set_title(
        "Figure S2. Fano-factor distribution across baseline settings",
        fontsize=12,
        fontweight="semibold",
    )
    _style_axes(ax)
    ax.legend(frameon=False)
    _add_caption_line(
        fig,
        f"Baseline PPP; K={int(bs['n_realizations'].median())}; frequency grid: 2-150 kHz (Δf = {int(bs['band_width_hz'].median())} Hz).",
    )
    out = paths.figures_dir / "figS2_fano_factor_by_setting.png"
    _save(fig, out)
    return out


def generate_all_figures(inputs: dict[str, Any], paths: config.Paths) -> list[Path]:
    outputs = [
        fig1_kernel_magnitude_vs_distance(paths),
        fig2_rms_vs_lambda(inputs, paths),
        fig3_p99_contour_lambda_R(inputs, paths),
        fig4_ccdf_ppp_vs_clustered(inputs, paths),
        fig5_phase_coherence_sweep(inputs, paths),
        fig6_inhomogeneous_same_mean_density(inputs, paths),
        fig7_predicted_vs_feeder_scatter(paths),
        fig8_allowable_lambda_vs_frequency(inputs, paths),
        figS1_denom_hist(inputs, paths),
        figS2_fano_distribution(inputs, paths),
    ]
    return outputs
