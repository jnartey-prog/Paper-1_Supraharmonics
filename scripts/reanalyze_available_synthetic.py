from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.ticker import FuncFormatter
from pandas.errors import EmptyDataError

try:
    from scripts.plotting_style import MANUSCRIPT_DPI, apply_manuscript_style, save_figure_bundle
except ModuleNotFoundError:  # pragma: no cover
    from plotting_style import MANUSCRIPT_DPI, apply_manuscript_style, save_figure_bundle

apply_manuscript_style()

REQ_SYN = {
    "analytical": "synthetic_latest_analytical.csv",
    "obs": "synthetic_latest_observations.csv",
    "samples": "synthetic_latest_per_frequency_samples.csv",
    "stats": "synthetic_latest_statistics.csv",
    "val": "synthetic_latest_validation.csv",
}
REQ_BENCH = {
    "perfreq": "validation_benchmark_per_frequency.csv",
    "summary": "validation_benchmark_summary.csv",
    "multi": "validation_benchmark_multiseed.csv",
    "multi_agg": "validation_benchmark_multiseed_aggregate.csv",
    "cfg": "validation_benchmark_run_config.json",
}
OPT_BENCH = {
    "geom_per_seed": "geometry_ablation_per_seed_metrics.csv",
    "geom_perfreq": "geometry_ablation_per_frequency.csv",
    "geom_table_g1": "table_g1_geometry_ablation_summary.csv",
    "geom_match": "geometry_ablation_match_report.csv",
    "geom_cfg": "geometry_ablation_run_config.json",
}
OPT = {
    "density": "synthetic_latest_density_sweep_statistics.csv",
    "density_multiseed": "synthetic_latest_density_sweep_multiseed.csv",
    "coherence": "synthetic_latest_coherence_sweep_statistics.csv",
    "coherence_multiseed": "synthetic_latest_coherence_sweep_multiseed.csv",
    "spatial": "synthetic_latest_spatial_process_statistics.csv",
    "spatial_multiseed": "synthetic_latest_spatial_process_multiseed.csv",
    "feeder_specs": "synthetic_latest_feeder_specs.csv",
    "feeder_kernels": "synthetic_latest_feeder_kernel_profiles.csv",
    "feeder_detail": "synthetic_latest_feeder_validation_detail.csv",
    "feeder_summary": "synthetic_latest_feeder_validation_summary.csv",
}


@dataclass(frozen=True)
class Inputs:
    analytical: pd.DataFrame
    obs: pd.DataFrame
    samples: pd.DataFrame
    stats: pd.DataFrame
    val: pd.DataFrame
    bench_pf: pd.DataFrame
    bench_sum: pd.DataFrame
    bench_multi: pd.DataFrame
    bench_multi_agg: pd.DataFrame
    bench_cfg: dict
    geom_per_seed: pd.DataFrame | None
    geom_perfreq: pd.DataFrame | None
    geom_table_g1: pd.DataFrame | None
    geom_match: pd.DataFrame | None
    geom_cfg: dict | None
    density: pd.DataFrame | None
    density_multiseed: pd.DataFrame | None
    coherence: pd.DataFrame | None
    coherence_multiseed: pd.DataFrame | None
    spatial: pd.DataFrame | None
    spatial_multiseed: pd.DataFrame | None
    feeder_specs: pd.DataFrame | None
    feeder_kernels: pd.DataFrame | None
    feeder_detail: pd.DataFrame | None
    feeder_summary: pd.DataFrame | None


def need_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{label}: missing columns {miss}")


def load_inputs(syn_dir: Path, bench_dir: Path, *, require_bench: bool = True) -> Inputs:
    sp = {k: syn_dir / v for k, v in REQ_SYN.items()}
    bp = {k: bench_dir / v for k, v in REQ_BENCH.items()}
    required_paths = list(sp.values()) + (list(bp.values()) if require_bench else [])
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    analytical = pd.read_csv(sp["analytical"])
    obs = pd.read_csv(sp["obs"])
    samples = pd.read_csv(sp["samples"])
    stats = pd.read_csv(sp["stats"])
    val = pd.read_csv(sp["val"])
    if require_bench:
        bench_pf = pd.read_csv(bp["perfreq"])
        bench_sum = pd.read_csv(bp["summary"])
        bench_multi = pd.read_csv(bp["multi"])
        bench_multi_agg = pd.read_csv(bp["multi_agg"])
        bench_cfg = json.loads(bp["cfg"].read_text(encoding="utf-8"))
    else:
        bench_pf = pd.DataFrame()
        bench_sum = pd.DataFrame()
        bench_multi = pd.DataFrame()
        bench_multi_agg = pd.DataFrame()
        bench_cfg = {}
    obp = {k: bench_dir / v for k, v in OPT_BENCH.items()}

    need_cols(
        stats,
        [
            "frequency_khz",
            "rms_abs_v",
            "p95_abs_v",
            "p99_abs_v",
            "exceedance_probability",
            "sample_size",
        ],
        "stats",
    )
    if require_bench:
        need_cols(
            bench_pf,
            ["benchmark_name", "frequency_khz", "relative_error_rms", "relative_error_p95"],
            "bench_pf",
        )

    op = {k: syn_dir / v for k, v in OPT.items()}

    def load(name: str) -> pd.DataFrame | None:
        path = op[name]
        return pd.read_csv(path) if path.exists() else None

    density = load("density")
    density_multiseed = load("density_multiseed")
    coherence = load("coherence")
    coherence_multiseed = load("coherence_multiseed")
    spatial = load("spatial")
    spatial_multiseed = load("spatial_multiseed")
    feeder_specs = load("feeder_specs")
    feeder_kernels = load("feeder_kernels")
    feeder_detail = load("feeder_detail")
    feeder_summary = load("feeder_summary")

    def _read_optional_csv(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return None

    geom_per_seed = _read_optional_csv(obp["geom_per_seed"])
    geom_perfreq = _read_optional_csv(obp["geom_perfreq"])
    geom_table_g1 = _read_optional_csv(obp["geom_table_g1"])
    geom_match = _read_optional_csv(obp["geom_match"])
    geom_cfg = (
        json.loads(obp["geom_cfg"].read_text(encoding="utf-8"))
        if obp["geom_cfg"].exists()
        else None
    )

    return Inputs(
        analytical,
        obs,
        samples,
        stats,
        val,
        bench_pf,
        bench_sum,
        bench_multi,
        bench_multi_agg,
        bench_cfg,
        geom_per_seed,
        geom_perfreq,
        geom_table_g1,
        geom_match,
        geom_cfg,
        density,
        density_multiseed,
        coherence,
        coherence_multiseed,
        spatial,
        spatial_multiseed,
        feeder_specs,
        feeder_kernels,
        feeder_detail,
        feeder_summary,
    )


def clear_out(
    out: Path, *, patterns: tuple[str, ...] = ("*.csv", "*.png", "*.pdf", "*.json")
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for p in out.glob(pat):
            try:
                p.unlink(missing_ok=True)
            except PermissionError:
                # Common on Windows if the user has a CSV open in Excel.
                continue


def save_csv(df: pd.DataFrame, out: Path, name: str) -> str:
    p = out / name
    df.to_csv(p, index=False)
    return str(p)


def save_fig(fig: plt.Figure, out: Path, name: str, *, dpi: int = MANUSCRIPT_DPI) -> str:
    p = out / name
    save_figure_bundle(fig, p, dpi=dpi)
    return str(p)


def style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25, linewidth=0.8)


def _add_figure_footer(
    fig: plt.Figure,
    lines: list[str],
    *,
    fontsize: int = 8,
    color: str = "#444444",
    bottom: float | None = None,
) -> None:
    bottom_margin = bottom if bottom is not None else (0.29 if len(lines) >= 2 else 0.22)
    fig.subplots_adjust(bottom=bottom_margin)
    y = 0.006
    for line in lines:
        fig.text(0.5, y, line, ha="center", va="bottom", fontsize=fontsize, color=color)
        y += 0.018


def _scenario_pretty(name: str) -> str:
    mapping = {
        "ppp": "PPP",
        "baseline_ppp": "PPP",
        "clustered": "Clustered",
        "clustered_thomas": "Clustered (Thomas)",
        "repulsive": "Repulsive",
        "inhomog_hotspot": "Inhomogeneous hotspot",
    }
    return mapping.get(str(name), str(name))


def _sci_density(x: float, _: float) -> str:
    if x <= 0.0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    coeff = x / (10**exp)
    if abs(coeff - 1.0) < 1e-12:
        return rf"$10^{{{exp}}}$"
    return rf"${coeff:.1f}\times10^{{{exp}}}$"


def _fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    lx = np.log10(np.maximum(x.astype(float), 1e-18))
    ly = np.log10(np.maximum(y.astype(float), 1e-18))
    if lx.size < 2:
        return float("nan")
    return float(np.polyfit(lx, ly, 1)[0])


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
        bbox=dict(facecolor="white", alpha=0.90, edgecolor="none", pad=0.2),
    )


def _make_appendix_monte_carlo_convergence_figure(i: Inputs, out: Path) -> str | None:
    required = {"frequency_khz", "sample_id", "abs_v"}
    if not required.issubset(i.samples.columns):
        return None
    samples = i.samples.copy()
    if samples.empty:
        return None

    target_frequency_khz = 30.0
    available_freqs = sorted(samples["frequency_khz"].unique().tolist())
    if not available_freqs:
        return None
    selected_frequency = min(available_freqs, key=lambda x: abs(float(x) - target_frequency_khz))
    sub = samples[np.isclose(samples["frequency_khz"], selected_frequency)].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("sample_id")
    values = sub["abs_v"].to_numpy(dtype=float)
    if values.size < 2:
        return None

    requested_k = [256, 512, 1024, 2048, 4096]
    k_values = [k for k in requested_k if k <= values.size]
    if not k_values:
        return None

    n_paths = 48
    rng = np.random.default_rng(20260220)
    rms_paths = np.full((n_paths, len(k_values)), np.nan, dtype=float)
    p99_paths = np.full((n_paths, len(k_values)), np.nan, dtype=float)
    for path_idx in range(n_paths):
        perm = rng.permutation(values.size)
        shuffled = values[perm]
        csum_sq = np.cumsum(shuffled**2)
        for j, k in enumerate(k_values):
            prefix = shuffled[:k]
            rms_paths[path_idx, j] = float(np.sqrt(csum_sq[k - 1] / k))
            p99_paths[path_idx, j] = float(np.percentile(prefix, 99.0))

    rms_med = np.nanmedian(rms_paths, axis=0)
    rms_lo = np.nanquantile(rms_paths, 0.025, axis=0)
    rms_hi = np.nanquantile(rms_paths, 0.975, axis=0)
    p99_med = np.nanmedian(p99_paths, axis=0)
    p99_lo = np.nanquantile(p99_paths, 0.025, axis=0)
    p99_hi = np.nanquantile(p99_paths, 0.975, axis=0)

    rms_full = float(np.sqrt(np.mean(values**2)))
    p99_full = float(np.percentile(values, 99.0))
    use_log_x = len(k_values) > 1
    xlim_left = max(1.0, float(min(k_values)) * 0.85)
    xlim_right = float(max(k_values)) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharex=True)
    manuscript_colors = {
        "rms": "#0B4F8A",
        "p99": "#B3570A",
        "reference": "#2D2D2D",
        "interval": 0.12,
    }
    axis_defs = [
        (
            axes[0],
            "RMS(|V|) [V]",
            rms_med,
            rms_lo,
            rms_hi,
            rms_full,
            "RMS estimate",
            "Full-sample RMS",
            manuscript_colors["rms"],
        ),
        (
            axes[1],
            "P99(|V|) [V]",
            p99_med,
            p99_lo,
            p99_hi,
            p99_full,
            "P99 estimate",
            "Full-sample P99",
            manuscript_colors["p99"],
        ),
    ]
    for idx, (ax, ylabel, med, lo, hi, ref, series_label, ref_label, color) in enumerate(axis_defs):
        ax.fill_between(
            k_values,
            lo,
            hi,
            color=color,
            alpha=manuscript_colors["interval"],
            linewidth=1.1,
            edgecolor=color,
            label="95% interval",
            zorder=1,
        )
        ax.plot(
            k_values,
            med,
            marker="o",
            lw=3.0,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7.0,
            label=series_label,
            zorder=3,
        )
        ax.axhline(
            ref,
            linestyle="--",
            lw=2.0,
            color=manuscript_colors["reference"],
            alpha=0.96,
            label=ref_label,
            zorder=2,
        )
        ax.set_ylabel(ylabel, fontsize=14, labelpad=8, fontweight="semibold")
        ax.set_xlabel("Monte Carlo realizations (K)", fontsize=12, labelpad=8, fontweight="semibold")
        ax.set_xticks(k_values)
        if use_log_x:
            ax.set_xscale("log", base=2)
        ax.set_xlim(xlim_left, xlim_right)
        ax.tick_params(axis="both", labelsize=12, width=1.3, length=5.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        style(ax)
        ax.legend(frameon=False, fontsize=11, loc="upper right", handlelength=2.2)
        _panel_label(ax, "(a)" if idx == 0 else "(b)")

    fig.subplots_adjust(bottom=0.30, top=0.84, wspace=0.20)
    fig.suptitle(
        f"Monte Carlo Convergence of RMS and P99 Estimates (reference frequency: {selected_frequency:.0f} kHz)",
        fontsize=17,
        fontweight="semibold",
    )
    return save_fig(fig, out, "figure_s1_monte_carlo_convergence_rms_p99.png", dpi=900)


def make_tables(i: Inputs, out: Path) -> dict[str, str]:
    gen: dict[str, str] = {}
    freqs = np.array(
        i.bench_cfg.get("frequencies_khz", sorted(i.stats["frequency_khz".strip()].unique())),
        dtype=float,
    )
    t1 = pd.DataFrame(
        [
            {
                "frequency_min_khz": float(freqs.min()),
                "frequency_max_khz": float(freqs.max()),
                "frequency_bins": int(freqs.size),
                "samples_per_frequency": int(i.stats["sample_size"].median()),
                "benchmark_variants": ",".join(sorted(i.bench_pf["benchmark_name"].unique())),
                "extended_assets": bool(
                    i.feeder_specs is not None
                    and i.feeder_kernels is not None
                    and i.feeder_detail is not None
                ),
            }
        ]
    )
    gen["table_1"] = save_csv(t1, out, "table_1_scenario_matrix.csv")

    t2 = pd.DataFrame(
        [
            {"metric": "mean", "column": "mean_abs_v"},
            {"metric": "variance", "column": "var_v"},
            {"metric": "rms", "column": "rms_abs_v"},
            {"metric": "p95", "column": "p95_abs_v"},
            {"metric": "p99", "column": "p99_abs_v"},
            {"metric": "exceedance", "column": "exceedance_probability"},
            {"metric": "benchmark_err_rms", "column": "relative_error_rms"},
            {"metric": "benchmark_err_p95", "column": "relative_error_p95"},
        ]
    )
    gen["table_2"] = save_csv(t2, out, "table_2_analytical_expressions.csv")

    t3 = (
        i.stats.merge(
            i.analytical[["frequency_khz", "rms_abs_v", "p95_abs_v"]].rename(
                columns={"rms_abs_v": "analytical_rms", "p95_abs_v": "analytical_p95"}
            ),
            on="frequency_khz",
            how="left",
        )
        .merge(
            i.val.rename(
                columns={
                    "relative_error_rms": "framework_vs_internal_relative_error_rms",
                    "relative_error_p95": "framework_vs_internal_relative_error_p95",
                }
            ),
            on="frequency_khz",
            how="left",
        )
        .sort_values("frequency_khz")
    )
    gen["table_3"] = save_csv(t3, out, "table_3_baseline_ppp_by_frequency.csv")

    worst = (
        i.bench_pf.assign(abs_r=i.bench_pf["relative_error_rms"].abs())
        .sort_values(["benchmark_name", "abs_r"], ascending=[True, False])
        .groupby("benchmark_name", as_index=False)
        .first()
    )
    t4 = i.bench_sum.merge(
        worst[["benchmark_name", "frequency_khz", "relative_error_rms", "relative_error_p95"]],
        on="benchmark_name",
        how="left",
    )
    gen["table_4"] = save_csv(t4, out, "table_4_sensitivity_summary.csv")

    if i.feeder_specs is not None and i.feeder_kernels is not None:
        rows = []
        for (fid, fname, f), g in i.feeder_kernels.groupby(
            ["feeder_id", "feeder_name", "frequency_khz"], sort=True
        ):
            g = g.sort_values("distance_m")

            def idx(distance_m: float) -> int:
                delta = np.abs(g["distance_m"].to_numpy(dtype=float) - distance_m)
                return int(np.argmin(delta))

            rows.append(
                {
                    "feeder_id": fid,
                    "feeder_name": fname,
                    "frequency_khz": f,
                    "ztr_mag_0m": float(g["ztr_mag_ohm"].iloc[idx(0.0)]),
                    "ztr_mag_100m": float(g["ztr_mag_ohm"].iloc[idx(100.0)]),
                    "ztr_mag_500m": float(g["ztr_mag_ohm"].iloc[idx(500.0)]),
                    "ztr_mag_1000m": float(g["ztr_mag_ohm"].iloc[idx(1000.0)]),
                }
            )
        t5 = pd.DataFrame(rows).merge(i.feeder_specs, on=["feeder_id", "feeder_name"], how="left")
    else:
        t5 = pd.DataFrame([{"status": "unavailable_from_inputs"}])
    gen["table_5"] = save_csv(t5, out, "table_5_feeder_benchmark_spec.csv")

    if i.feeder_detail is not None:
        d = i.feeder_detail
        overall = pd.DataFrame(
            [
                {
                    "benchmark_name": "framework_vs_synthetic_feeder_overall",
                    "n_frequencies": int(d["frequency_khz"].nunique()),
                    "rms_error_mean": float(d["relative_error_rms"].mean()),
                    "rms_error_p90": float(d["relative_error_rms"].quantile(0.9)),
                    "p95_error_mean": float(d["relative_error_p95"].mean()),
                    "p95_error_p90": float(d["relative_error_p95"].quantile(0.9)),
                    "p99_error_mean": float(d["relative_error_p99"].mean()),
                    "p99_error_p90": float(d["relative_error_p99"].quantile(0.9)),
                }
            ]
        )
        if i.feeder_summary is not None:
            fs = i.feeder_summary.copy()
            fs["benchmark_name"] = fs["feeder_id"].map(
                lambda x: f"framework_vs_synthetic_feeder_{x}"
            )
            t6 = pd.concat([overall, fs], ignore_index=True, sort=False)
        else:
            t6 = overall
    else:
        t6 = i.bench_sum.copy()
    gen["table_6"] = save_csv(t6, out, "table_6_analytical_vs_feeder_metrics.csv")
    if i.geom_table_g1 is not None and not i.geom_table_g1.empty:
        gen["table_g1"] = save_csv(i.geom_table_g1, out, "table_g1_geometry_ablation_summary.csv")
    elif i.geom_perfreq is not None and not i.geom_perfreq.empty:
        g = i.geom_perfreq.copy()
        rows = []
        for (fid, fname, density, geom), sub in g.groupby(
            ["feeder_id", "feeder_name", "density", "geometry"], sort=True
        ):
            sub = sub.sort_values("frequency_khz")
            row = {
                "feeder_id": fid,
                "feeder_name": fname,
                "density": float(density),
                "geometry": geom,
            }
            for metric in ("rms", "p95", "p99"):
                col = f"rel_diff_{metric}"
                vals = sub[col].to_numpy(dtype=float)
                row[f"{metric}_rel_diff_mean"] = float(np.mean(vals))
                row[f"{metric}_rel_diff_p90"] = float(np.quantile(vals, 0.9))
                row[f"{metric}_rel_diff_max"] = float(np.max(vals))
                row[f"{metric}_worst_frequency_khz"] = float(
                    sub.iloc[int(np.argmax(vals))]["frequency_khz"]
                )
            rows.append(row)
        gen["table_g1"] = save_csv(
            pd.DataFrame(rows), out, "table_g1_geometry_ablation_summary.csv"
        )
    return gen


def make_figures(i: Inputs, out: Path) -> dict[str, str]:
    g: dict[str, str] = {}
    if i.geom_perfreq is not None and not i.geom_perfreq.empty:
        gp = i.geom_perfreq.copy()
        rep_density = None
        if (
            i.geom_cfg
            and "representative_density" in i.geom_cfg
            and i.geom_cfg["representative_density"] is not None
        ):
            rep_density = float(i.geom_cfg["representative_density"])
        if rep_density is None:
            rep_density = float(np.median(sorted(gp["density"].unique().tolist())))
        feeder_candidates = sorted(gp["feeder_id"].astype(str).unique().tolist())
        rep_feeder = "B" if "B" in feeder_candidates else feeder_candidates[0]
        sub = gp[
            (gp["feeder_id"].astype(str) == rep_feeder) & np.isclose(gp["density"], rep_density)
        ].sort_values("frequency_khz")
        if not sub.empty:
            fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True)
            metric_defs = [
                ("rms", "RMS"),
                ("p95", "P95"),
                ("p99", "P99"),
            ]
            for ax, (metric, label) in zip(axes, metric_defs):
                ax.plot(sub["frequency_khz"], sub[f"{metric}_abs_v_disc"], lw=1.9, label="disc")
                geom_col = [
                    c
                    for c in sub.columns
                    if c.startswith(f"{metric}_abs_v_") and c != f"{metric}_abs_v_disc"
                ]
                if geom_col:
                    ax.plot(sub["frequency_khz"], sub[geom_col[0]], lw=1.9, label="branched")
                ax2 = ax.twinx()
                ax2.plot(
                    sub["frequency_khz"],
                    100.0 * sub[f"rel_diff_{metric}"],
                    color="#d62728",
                    alpha=0.55,
                    lw=1.1,
                )
                ax.set_ylabel(f"{label} [V]")
                ax2.set_ylabel("rel diff [%]", color="#d62728")
                style(ax)
            axes[0].legend(frameon=False, ncol=2)
            axes[-1].set_xlabel("Frequency [kHz]")
            fig.suptitle(
                f"Figure G1: Disc vs branched, feeder {rep_feeder}, density={rep_density:g}"
            )
            g["figure_g1"] = save_fig(fig, out, "figure_g1_disc_vs_branched.png")

            table6_ref = None
            if i.feeder_summary is not None and "p99_error_mean" in i.feeder_summary.columns:
                table6_ref = float(i.feeder_summary["p99_error_mean"].mean())
            elif i.feeder_detail is not None and "relative_error_p99" in i.feeder_detail.columns:
                table6_ref = float(i.feeder_detail["relative_error_p99"].mean())
            if table6_ref is None and i.geom_table_g1 is not None and not i.geom_table_g1.empty:
                table6_ref = float(i.geom_table_g1["p99_rel_diff_mean"].mean())
            if table6_ref is not None:
                if i.geom_table_g1 is not None and not i.geom_table_g1.empty:
                    tg = i.geom_table_g1.copy()
                else:
                    tg = (
                        gp.groupby(["feeder_id", "density"], as_index=False)["rel_diff_p99"]
                        .mean()
                        .rename(columns={"rel_diff_p99": "p99_rel_diff_mean"})
                    )
                labels = tg.apply(
                    lambda r: f"F{r['feeder_id']}, d={float(r['density']):g}", axis=1
                ).tolist()
                vals = 100.0 * tg["p99_rel_diff_mean"].to_numpy(dtype=float)
                ref = 100.0 * table6_ref
                fig, ax = plt.subplots(figsize=(10.5, 4.8))
                ax.bar(np.arange(len(vals)), vals, label="geometry-only p99 rel diff")
                ax.axhline(
                    ref, color="#d62728", linestyle="--", lw=1.4, label="Table 6 residual scale"
                )
                ax.set_xticks(np.arange(len(vals)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel("Relative difference [%]")
                ax.set_title("Figure G2: Geometry-only error vs Table 6 residual")
                style(ax)
                ax.legend(frameon=False)
                g["figure_g2"] = save_fig(fig, out, "figure_g2_geometry_vs_table6_residual.png")

    if i.feeder_kernels is not None:
        k = i.feeder_kernels
        target = [10.0, 30.0, 80.0, 150.0]
        av = sorted(k["frequency_khz"].unique().tolist())
        sel = [min(av, key=lambda x: abs(x - t)) for t in target]
        feeders = sorted(k["feeder_id"].astype(str).unique().tolist())
        palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
        fig, axes_abs = plt.subplots(1, 3, figsize=(7.6, 4.2), sharey=True)
        fig._skip_tight_layout = True
        fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.36, wspace=0.30)
        for ax_abs, fid in zip(axes_abs, feeders):
            sub = k[k["feeder_id"].astype(str) == fid]
            for color, f in zip(palette, sel):
                sf = sub[np.isclose(sub["frequency_khz"], f)].sort_values("distance_m")
                x = sf["distance_m"].to_numpy(dtype=float)
                y = sf["ztr_mag_ohm"].to_numpy(dtype=float)
                ax_abs.plot(x, y, color=color, lw=2.6, label=f"{f:.0f} kHz", solid_capstyle="round")
            ax_abs.set_title(f"Feeder {fid} kernel", fontsize=12, fontweight="semibold")
            ax_abs.set_xlabel("Distance [m]", fontsize=12, labelpad=6)
            style(ax_abs)
            ax_abs.tick_params(labelsize=11)
            ax_abs.text(
                0.03,
                0.93,
                "Synthetic model: |Ztr(f,d)|",
                transform=ax_abs.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        # Add a small headroom to avoid thick lines looking clipped at the top.
        all_y: list[np.ndarray] = []
        for ax_abs in axes_abs:
            for line in ax_abs.get_lines():
                yd = np.asarray(line.get_ydata(), dtype=float)
                if yd.size:
                    all_y.append(yd)
        if all_y:
            y = np.concatenate(all_y)
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                pad = 0.08 * (y_max - y_min)
                axes_abs[0].set_ylim(y_min - 0.03 * (y_max - y_min), y_max + pad)
        axes_abs[0].set_ylabel("|Ztr| [Ω]", fontsize=12)
        handles, labels = axes_abs[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.16),
            ncol=4,
            frameon=False,
            fontsize=11,
            handlelength=3.2,
            columnspacing=1.6,
        )
        fig.suptitle(
            "Figure 1: Transfer-impedance kernel magnitude versus distance",
            fontsize=13,
            fontweight="semibold",
            y=0.98,
        )
        g["figure_1"] = save_fig(fig, out, "figure_1_transfer_impedance_vs_distance.png")

    if i.density is not None:
        d = i.density
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        palette = ["#0072B2", "#E69F00", "#009E73"]
        slopes: list[float] = []
        has_density_bands = False
        density_band_seed_count = None
        ref_line_x: np.ndarray | None = None
        ref_line_y: np.ndarray | None = None
        for idx, t in enumerate([10.0, 30.0, 100.0]):
            f = min(sorted(d["frequency_khz"].unique().tolist()), key=lambda x: abs(x - t))
            sub = d[np.isclose(d["frequency_khz"], f)].sort_values("density")
            x_vals = sub["density"].to_numpy(dtype=float)
            y_vals = sub["rms_abs_v"].to_numpy(dtype=float)
            if i.density_multiseed is not None and {
                "frequency_khz",
                "density",
                "seed",
                "rms_abs_v",
            }.issubset(i.density_multiseed.columns):
                ms = i.density_multiseed[np.isclose(i.density_multiseed["frequency_khz"], f)].copy()
                if not ms.empty:
                    density_band_seed_count = int(ms["seed"].nunique())
                    q = (
                        ms.groupby("density")["rms_abs_v"]
                        .quantile([0.025, 0.5, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.5: "q500", 0.975: "q975"})
                        .reset_index()
                        .sort_values("density")
                    )
                    if not q.empty:
                        x_vals = q["density"].to_numpy(dtype=float)
                        y_vals = q["q500"].to_numpy(dtype=float)
            (line,) = ax.loglog(
                x_vals,
                y_vals,
                marker="o",
                lw=2.6,
                markersize=7.0,
                color=palette[idx % len(palette)],
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{f:.0f} kHz",
            )
            if i.density_multiseed is not None and {
                "frequency_khz",
                "density",
                "seed",
                "rms_abs_v",
            }.issubset(i.density_multiseed.columns):
                ms = i.density_multiseed[np.isclose(i.density_multiseed["frequency_khz"], f)].copy()
                if not ms.empty:
                    density_band_seed_count = int(ms["seed"].nunique())
                    q = (
                        ms.groupby("density")["rms_abs_v"]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("density")
                    )
                    if not q.empty:
                        ax.fill_between(
                            q["density"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=line.get_color(),
                            alpha=0.14,
                            linewidth=0.0,
                        )
                        has_density_bands = True
            slope = _fit_loglog_slope(
                x_vals,
                y_vals,
            )
            if np.isfinite(slope):
                slopes.append(slope)
            if int(round(f)) == 30:
                ref_line_x = x_vals
                ref_line_y = y_vals
        if slopes:
            if ref_line_x is not None and ref_line_y is not None and ref_line_x.size > 0:
                ref_line = ref_line_y[0] * np.sqrt(ref_line_x / max(ref_line_x[0], 1e-18))
                ax.loglog(
                    ref_line_x,
                    ref_line,
                    linestyle="--",
                    lw=1.9,
                    color="#333333",
                    alpha=0.95,
                    label="sqrt(density) reference",
                )
            ax.text(
                0.02,
                0.96,
                f"Fitted log-log slopes: {', '.join(f'{s:.2f}' for s in slopes)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        if has_density_bands:
            nseeds = density_band_seed_count if density_band_seed_count is not None else 5
            ax.text(
                0.02,
                0.88,
                f"Shaded bands: 95% across {nseeds} seeds",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        ax.set_xlabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax.set_ylabel("RMS(|V|) [V]", fontsize=13, labelpad=6)
        ax.set_title("Figure 2: RMS vs density", fontsize=14, fontweight="semibold", pad=10)
        ax.xaxis.set_major_formatter(FuncFormatter(_sci_density))
        ax.tick_params(labelsize=12)
        ax.legend(frameon=False, fontsize=12)
        style(ax)
        ax.grid(alpha=0.30, linewidth=1.0)
        # Keep Figure 2 clean; dataset/method notes belong in manuscript caption text.
        g["figure_2"] = save_fig(fig, out, "figure_2_rms_vs_density.png")

        piv_rms = d.pivot(index="frequency_khz", columns="density", values="rms_abs_v").sort_index()
        x = piv_rms.columns.to_numpy(dtype=float)
        y = piv_rms.index.to_numpy(dtype=float)
        z = piv_rms.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        im = ax.pcolormesh(x, y, z, shading="nearest", cmap="viridis")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], rotation=35, ha="right")
        y_ticks = np.linspace(y.min(), y.max(), 8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v:.0f}" for v in y_ticks])
        ax.set_xlabel("Density [scenario units]")
        ax.set_ylabel("Frequency [kHz]")
        ax.set_title("Figure 2b: full-band RMS map")
        contour = ax.contour(
            x,
            y,
            z,
            colors="white",
            linewidths=0.8,
            alpha=0.7,
        )
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
        fig.colorbar(im, ax=ax, label="RMS(|V|) [V]")
        # Keep Figure 2b clean; dataset/method notes belong in manuscript caption text.
        g["figure_2b"] = save_fig(fig, out, "figure_2b_rms_fullband_heatmap.png")

        piv = d.pivot(index="frequency_khz", columns="density", values="p99_abs_v").sort_index()
        x = piv.columns.to_numpy(dtype=float)
        y = piv.index.to_numpy(dtype=float)
        z = piv.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        im = ax.pcolormesh(x, y, z, shading="nearest", cmap="viridis")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], rotation=30, ha="right")
        y_ticks = np.linspace(y.min(), y.max(), 8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v:.0f}" for v in y_ticks])
        ax.set_xlabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax.set_ylabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax.set_title("Figure 3: P99 Map", fontsize=14, fontweight="semibold", pad=10)
        ax.tick_params(labelsize=12)
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        candidate_levels = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56], dtype=float)
        levels = candidate_levels[(candidate_levels >= z_min) & (candidate_levels <= z_max)]
        contour = ax.contour(
            x,
            y,
            z,
            levels=levels if levels.size else None,
            colors="white",
            linewidths=1.2,
            alpha=0.90,
        )
        # Outline contours so they remain visible across both dark and light regions.
        contour.set_path_effects(
            [pe.Stroke(linewidth=2.2, foreground="black", alpha=0.55), pe.Normal()]
        )
        labels = ax.clabel(contour, inline=True, inline_spacing=3, fontsize=12, fmt="%.2f")
        for t in labels:
            t.set_path_effects(
                [pe.Stroke(linewidth=2.0, foreground="black", alpha=0.65), pe.Normal()]
            )
            t.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.2))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("P99(|V|) [V]", fontsize=13, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        # Keep Figure 3 clean; dataset/method notes belong in manuscript caption text.
        g["figure_3"] = save_fig(fig, out, "figure_3_percentile_design_curves.png")

        # Composite (Word): Figure 2 + Figure 3 (keep originals as well).
        fig = plt.figure(figsize=(10.8, 4.8))
        fig._skip_tight_layout = True
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.06], wspace=0.22)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        # (a) RMS vs density
        palette = ["#0072B2", "#E69F00", "#009E73"]
        slopes: list[float] = []
        has_density_bands = False
        density_band_seed_count = None
        ref_line_x: np.ndarray | None = None
        ref_line_y: np.ndarray | None = None
        for idx, t in enumerate([10.0, 30.0, 100.0]):
            f = min(sorted(d["frequency_khz"].unique().tolist()), key=lambda x: abs(x - t))
            sub = d[np.isclose(d["frequency_khz"], f)].sort_values("density")
            x_vals = sub["density"].to_numpy(dtype=float)
            y_vals = sub["rms_abs_v"].to_numpy(dtype=float)
            if i.density_multiseed is not None and {
                "frequency_khz",
                "density",
                "seed",
                "rms_abs_v",
            }.issubset(i.density_multiseed.columns):
                ms = i.density_multiseed[np.isclose(i.density_multiseed["frequency_khz"], f)].copy()
                if not ms.empty:
                    density_band_seed_count = int(ms["seed"].nunique())
                    q = (
                        ms.groupby("density")["rms_abs_v"]
                        .quantile([0.025, 0.5, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.5: "q500", 0.975: "q975"})
                        .reset_index()
                        .sort_values("density")
                    )
                    if not q.empty:
                        x_vals = q["density"].to_numpy(dtype=float)
                        y_vals = q["q500"].to_numpy(dtype=float)
            (line,) = ax_a.loglog(
                x_vals,
                y_vals,
                marker="o",
                lw=2.6,
                markersize=7.0,
                color=palette[idx % len(palette)],
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{f:.0f} kHz",
            )
            if i.density_multiseed is not None and {
                "frequency_khz",
                "density",
                "seed",
                "rms_abs_v",
            }.issubset(i.density_multiseed.columns):
                ms = i.density_multiseed[np.isclose(i.density_multiseed["frequency_khz"], f)].copy()
                if not ms.empty:
                    density_band_seed_count = int(ms["seed"].nunique())
                    q = (
                        ms.groupby("density")["rms_abs_v"]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("density")
                    )
                    if not q.empty:
                        ax_a.fill_between(
                            q["density"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=line.get_color(),
                            alpha=0.14,
                            linewidth=0.0,
                        )
                        has_density_bands = True
            slope = _fit_loglog_slope(x_vals, y_vals)
            if np.isfinite(slope):
                slopes.append(slope)
            if int(round(f)) == 30:
                ref_line_x = x_vals
                ref_line_y = y_vals
        if slopes and ref_line_x is not None and ref_line_y is not None and ref_line_x.size > 0:
            ref_line = ref_line_y[0] * np.sqrt(ref_line_x / max(ref_line_x[0], 1e-18))
            ax_a.loglog(
                ref_line_x,
                ref_line,
                linestyle="--",
                lw=1.9,
                color="#333333",
                alpha=0.95,
                label="sqrt(density) reference",
            )
        if slopes:
            ax_a.text(
                0.02,
                0.96,
                f"Fitted log-log slopes: {', '.join(f'{s:.2f}' for s in slopes)}",
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        if has_density_bands:
            nseeds = density_band_seed_count if density_band_seed_count is not None else 5
            ax_a.text(
                0.02,
                0.88,
                f"Shaded bands: 95% across {nseeds} seeds",
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        ax_a.set_xlabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax_a.set_ylabel("RMS(|V|) [V]", fontsize=13, labelpad=6)
        ax_a.set_title("RMS vs density", fontsize=13, fontweight="semibold", pad=8)
        ax_a.xaxis.set_major_formatter(FuncFormatter(_sci_density))
        ax_a.tick_params(labelsize=12)
        ax_a.legend(frameon=False, fontsize=12)
        style(ax_a)
        ax_a.grid(alpha=0.30, linewidth=1.0)
        _panel_label(ax_a, "(a)")

        # (b) P99 map
        piv = d.pivot(index="frequency_khz", columns="density", values="p99_abs_v").sort_index()
        x = piv.columns.to_numpy(dtype=float)
        y = piv.index.to_numpy(dtype=float)
        z = piv.to_numpy(dtype=float)
        im = ax_b.pcolormesh(x, y, z, shading="nearest", cmap="viridis")
        ax_b.set_xscale("log")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels([f"{v:g}" for v in x], rotation=30, ha="right")
        y_ticks = np.linspace(y.min(), y.max(), 8)
        ax_b.set_yticks(y_ticks)
        ax_b.set_yticklabels([f"{v:.0f}" for v in y_ticks])
        ax_b.set_xlabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax_b.set_ylabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax_b.set_title("P99 map", fontsize=13, fontweight="semibold", pad=8)
        ax_b.tick_params(labelsize=12)
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        candidate_levels = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56], dtype=float)
        levels = candidate_levels[(candidate_levels >= z_min) & (candidate_levels <= z_max)]
        contour = ax_b.contour(
            x,
            y,
            z,
            levels=levels if levels.size else None,
            colors="white",
            linewidths=1.2,
            alpha=0.90,
        )
        contour.set_path_effects(
            [pe.Stroke(linewidth=2.2, foreground="black", alpha=0.55), pe.Normal()]
        )
        labels = ax_b.clabel(contour, inline=True, inline_spacing=3, fontsize=12, fmt="%.2f")
        for t in labels:
            t.set_path_effects(
                [pe.Stroke(linewidth=2.0, foreground="black", alpha=0.65), pe.Normal()]
            )
            t.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.2))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("P99(|V|) [V]", fontsize=13, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        _panel_label(ax_b, "(b)")

        g["figure_2_3_composite"] = save_fig(fig, out, "figure_2_3_composite.png")
    else:
        s = i.stats.sort_values("frequency_khz")
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.plot(s["frequency_khz"], s["rms_abs_v"], lw=2.0)
        ax.set_title("Figure 2 fallback")
        style(ax)
        g["figure_2"] = save_fig(fig, out, "figure_2_rms_vs_density.png")

    if i.spatial is not None:
        sp = i.spatial
        f = min(sorted(sp["frequency_khz"].unique().tolist()), key=lambda x: abs(x - 30.0))
        sub = sp[np.isclose(sp["frequency_khz"], f)].copy()
        sub["scenario_pretty"] = sub["scenario_name"].map(_scenario_pretty)
        sub = sub.sort_values("scenario_pretty")
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        y_vals = sub["p99_abs_v"].to_numpy(dtype=float)
        yerr = None
        seed_count = None
        if i.spatial_multiseed is not None and {
            "scenario_name",
            "frequency_khz",
            "seed",
            "p99_abs_v",
        }.issubset(i.spatial_multiseed.columns):
            ms = i.spatial_multiseed[np.isclose(i.spatial_multiseed["frequency_khz"], f)].copy()
            if not ms.empty:
                agg = (
                    ms.groupby("scenario_name")["p99_abs_v"]
                    .agg(
                        [
                            "mean",
                            lambda s: np.quantile(s.to_numpy(dtype=float), 0.025),
                            lambda s: np.quantile(s.to_numpy(dtype=float), 0.975),
                            "count",
                        ]
                    )
                    .rename(
                        columns={
                            "<lambda_0>": "q025",
                            "<lambda_1>": "q975",
                            "mean": "mean",
                            "count": "n_seed",
                        }
                    )
                    .reset_index()
                )
                sub = sub.merge(agg, on="scenario_name", how="left")
                if sub["mean"].notna().all():
                    y_vals = sub["mean"].to_numpy(dtype=float)
                    lo = sub["q025"].to_numpy(dtype=float)
                    hi = sub["q975"].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y_vals - lo, 0.0), np.maximum(hi - y_vals, 0.0)])
                    seed_count = int(np.nanmedian(sub["n_seed"].to_numpy(dtype=float)))
        bars = ax.bar(
            sub["scenario_pretty"],
            y_vals,
            yerr=yerr,
            capsize=4.5 if yerr is not None else 0.0,
            color=["#0072B2", "#E69F00", "#009E73"][: len(sub)],
            edgecolor="#1a1a1a",
            linewidth=0.8,
        )
        if yerr is not None:
            for cap in ax.lines:
                cap.set_linewidth(1.6)
        if "ppp" in sub["scenario_name"].values:
            ppp_idx = int(np.where(sub["scenario_name"].to_numpy(dtype=str) == "ppp")[0][0])
            ppp_val = float(y_vals[ppp_idx])
            for idx_bar, (bar, val) in enumerate(zip(bars, y_vals)):
                delta = 100.0 * (val - ppp_val) / max(ppp_val, 1e-12)
                y_pad = 0.012 * max(float(np.nanmax(y_vals)), 1e-6)
                if yerr is not None:
                    y_top = val + float(yerr[1, idx_bar]) + y_pad
                else:
                    y_top = bar.get_height() * 1.01 + y_pad
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0 + 0.01 * bar.get_width(),
                    y_top,
                    f"{delta:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="semibold",
                )
        ax.set_title(
            f"Figure 4: Spatial-Process Tails at {f:.0f} kHz",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        ax.set_ylabel("P99(|V|) [V]", fontsize=13, labelpad=6)
        ax.set_xlabel("Spatial process", fontsize=13, labelpad=6)
        ax.tick_params(labelsize=12)
        if seed_count is not None:
            ax.text(
                0.02,
                0.95,
                f"Error bars: 95% interval across {seed_count} seeds",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        style(ax)
        ax.grid(alpha=0.30, linewidth=1.0)
        # Keep Figure 4 clean; dataset/method notes belong in manuscript caption text.
        g["figure_4"] = save_fig(fig, out, "figure_4_ccdf_ppp_cluster_repulsive.png")

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
        palette = {"ppp": "#E69F00", "clustered": "#0072B2", "repulsive": "#009E73"}
        p = sp.pivot(
            index="frequency_khz", columns="scenario_name", values="p99_abs_v"
        ).sort_index()
        ms_sp = None
        if i.spatial_multiseed is not None and {
            "scenario_name",
            "frequency_khz",
            "seed",
            "p99_abs_v",
        }.issubset(i.spatial_multiseed.columns):
            ms_sp = i.spatial_multiseed.copy()
            p_med = (
                ms_sp.groupby(["frequency_khz", "scenario_name"])["p99_abs_v"]
                .median()
                .reset_index()
                .pivot(index="frequency_khz", columns="scenario_name", values="p99_abs_v")
                .sort_index()
            )
            if not p_med.empty:
                p = p_med
        has_spatial_bands = False
        spatial_band_seed_count = None
        for c in p.columns:
            c_str = str(c)
            axes[0].plot(
                p.index,
                p[c],
                lw=2.6,
                color=palette.get(c_str, None),
                label=_scenario_pretty(c_str),
            )
            if ms_sp is not None:
                ms_c = ms_sp[ms_sp["scenario_name"].astype(str) == c_str]
                if not ms_c.empty:
                    spatial_band_seed_count = int(ms_c["seed"].nunique())
                    q = (
                        ms_c.groupby("frequency_khz")["p99_abs_v"]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("frequency_khz")
                    )
                    if not q.empty:
                        axes[0].fill_between(
                            q["frequency_khz"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=palette.get(c_str, None),
                            alpha=0.12,
                            linewidth=0.0,
                        )
                        has_spatial_bands = True
        ex_col = (
            "exceedance_probability_fixed_tau03"
            if "exceedance_probability_fixed_tau03" in sp.columns
            else "exceedance_probability"
        )
        tau_txt = ""
        if ex_col == "exceedance_probability_fixed_tau03":
            tau_v = (
                float(sp["exceedance_tau_fixed_v"].iloc[0])
                if "exceedance_tau_fixed_v" in sp.columns
                else 0.30
            )
            tau_txt = f" (|V|>{tau_v:.2f} V)"
        e = sp.pivot(index="frequency_khz", columns="scenario_name", values=ex_col).sort_index()
        ex_col_ms = (
            ex_col if (ms_sp is not None and ex_col in ms_sp.columns) else "exceedance_probability"
        )
        if ms_sp is not None and ex_col_ms in ms_sp.columns:
            e_med = (
                ms_sp.groupby(["frequency_khz", "scenario_name"])[ex_col_ms]
                .median()
                .reset_index()
                .pivot(index="frequency_khz", columns="scenario_name", values=ex_col_ms)
                .sort_index()
            )
            if not e_med.empty:
                e = e_med
        for c in e.columns:
            c_str = str(c)
            axes[1].plot(
                e.index,
                e[c],
                lw=2.6,
                color=palette.get(c_str, None),
                label=_scenario_pretty(c_str),
            )
            if ms_sp is not None and ex_col_ms in ms_sp.columns:
                ms_c = ms_sp[ms_sp["scenario_name"].astype(str) == c_str]
                if not ms_c.empty:
                    q = (
                        ms_c.groupby("frequency_khz")[ex_col_ms]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("frequency_khz")
                    )
                    if not q.empty:
                        axes[1].fill_between(
                            q["frequency_khz"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=palette.get(c_str, None),
                            alpha=0.12,
                            linewidth=0.0,
                        )
                        has_spatial_bands = True
        axes[0].legend(frameon=False, fontsize=12)
        axes[0].set_ylabel("P99(|V|) [V]", fontsize=13, labelpad=6)
        axes[1].set_ylabel(f"Exceedance probability{tau_txt}", fontsize=13, labelpad=8)
        if ex_col == "exceedance_probability_fixed_tau03":
            axes[1].text(
                0.02,
                0.95,
                "Fixed screening threshold",
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        if has_spatial_bands:
            nseeds = spatial_band_seed_count if spatial_band_seed_count is not None else 5
            axes[0].text(
                0.02,
                0.86,
                f"Shaded bands: 95% across {nseeds} seeds",
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        axes[0].set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        axes[1].set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        axes[0].tick_params(labelsize=12)
        axes[1].tick_params(labelsize=12)
        style(axes[0])
        style(axes[1])
        axes[0].grid(alpha=0.30, linewidth=1.0)
        axes[1].grid(alpha=0.30, linewidth=1.0)
        fig.suptitle("Figure 6: Process Effects", fontsize=14, fontweight="semibold", y=0.98)
        g["figure_6"] = save_fig(fig, out, "figure_6_inhomogeneous_intensity_outcomes.png")

        # Composite (Word): Figure 4 + Figure 6 (keep originals as well).
        fig = plt.figure(figsize=(10.8, 7.0))
        fig._skip_tight_layout = True
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.30)
        ax_a = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[1, 1])

        # (a) Spatial-process tails @ representative frequency
        f0 = min(sorted(sp["frequency_khz"].unique().tolist()), key=lambda x: abs(x - 30.0))
        sub = sp[np.isclose(sp["frequency_khz"], f0)].copy()
        sub["scenario_pretty"] = sub["scenario_name"].map(_scenario_pretty)
        sub = sub.sort_values("scenario_pretty")
        y_vals = sub["p99_abs_v"].to_numpy(dtype=float)
        yerr = None
        if i.spatial_multiseed is not None and {
            "scenario_name",
            "frequency_khz",
            "seed",
            "p99_abs_v",
        }.issubset(i.spatial_multiseed.columns):
            ms = i.spatial_multiseed[np.isclose(i.spatial_multiseed["frequency_khz"], f0)].copy()
            if not ms.empty:
                agg = (
                    ms.groupby("scenario_name")["p99_abs_v"]
                    .agg(
                        [
                            "mean",
                            lambda s: np.quantile(s.to_numpy(dtype=float), 0.025),
                            lambda s: np.quantile(s.to_numpy(dtype=float), 0.975),
                        ]
                    )
                    .rename(columns={"<lambda_0>": "q025", "<lambda_1>": "q975", "mean": "mean"})
                    .reset_index()
                )
                sub2 = sub.merge(agg, on="scenario_name", how="left")
                if sub2["mean"].notna().all():
                    y_vals = sub2["mean"].to_numpy(dtype=float)
                    lo = sub2["q025"].to_numpy(dtype=float)
                    hi = sub2["q975"].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y_vals - lo, 0.0), np.maximum(hi - y_vals, 0.0)])
        bars = ax_a.bar(
            sub["scenario_pretty"],
            y_vals,
            yerr=yerr,
            capsize=4.5 if yerr is not None else 0.0,
            color=["#0072B2", "#E69F00", "#009E73"][: len(sub)],
            edgecolor="#1a1a1a",
            linewidth=0.8,
        )
        if "ppp" in sub["scenario_name"].values:
            ppp_idx = int(np.where(sub["scenario_name"].to_numpy(dtype=str) == "ppp")[0][0])
            ppp_val = float(y_vals[ppp_idx])
            for idx_bar, (bar, val) in enumerate(zip(bars, y_vals)):
                delta = 100.0 * (val - ppp_val) / max(ppp_val, 1e-12)
                y_pad = 0.012 * max(float(np.nanmax(y_vals)), 1e-6)
                if yerr is not None:
                    y_top = val + float(yerr[1, idx_bar]) + y_pad
                else:
                    y_top = bar.get_height() * 1.01 + y_pad
                ax_a.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_top,
                    f"{delta:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="semibold",
                )
        ax_a.set_title(
            f"Spatial-process tails at {f0:.0f} kHz", fontsize=13, fontweight="semibold", pad=8
        )
        ax_a.set_ylabel("P99(|V|) [V]", fontsize=13, labelpad=6)
        ax_a.set_xlabel("Spatial process", fontsize=13, labelpad=6)
        ax_a.tick_params(labelsize=12)
        style(ax_a)
        ax_a.grid(alpha=0.30, linewidth=1.0)
        _panel_label(ax_a, "(a)")

        # (b,c) Frequency sweeps (reuse the same pivots as Figure 6)
        palette = {"ppp": "#E69F00", "clustered": "#0072B2", "repulsive": "#009E73"}
        p = sp.pivot(
            index="frequency_khz", columns="scenario_name", values="p99_abs_v"
        ).sort_index()
        ms_sp = None
        if i.spatial_multiseed is not None and {
            "scenario_name",
            "frequency_khz",
            "seed",
            "p99_abs_v",
        }.issubset(i.spatial_multiseed.columns):
            ms_sp = i.spatial_multiseed.copy()
            p_med = (
                ms_sp.groupby(["frequency_khz", "scenario_name"])["p99_abs_v"]
                .median()
                .reset_index()
                .pivot(index="frequency_khz", columns="scenario_name", values="p99_abs_v")
                .sort_index()
            )
            if not p_med.empty:
                p = p_med
        has_spatial_bands = False
        spatial_band_seed_count = None
        for c in p.columns:
            c_str = str(c)
            ax_b.plot(
                p.index, p[c], lw=2.6, color=palette.get(c_str, None), label=_scenario_pretty(c_str)
            )
            if ms_sp is not None:
                ms_c = ms_sp[ms_sp["scenario_name"].astype(str) == c_str]
                if not ms_c.empty:
                    spatial_band_seed_count = int(ms_c["seed"].nunique())
                    q = (
                        ms_c.groupby("frequency_khz")["p99_abs_v"]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("frequency_khz")
                    )
                    if not q.empty:
                        ax_b.fill_between(
                            q["frequency_khz"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=palette.get(c_str, None),
                            alpha=0.12,
                            linewidth=0.0,
                        )
                        has_spatial_bands = True
        ex_col = (
            "exceedance_probability_fixed_tau03"
            if "exceedance_probability_fixed_tau03" in sp.columns
            else "exceedance_probability"
        )
        tau_txt = ""
        if ex_col == "exceedance_probability_fixed_tau03":
            tau_v = (
                float(sp["exceedance_tau_fixed_v"].iloc[0])
                if "exceedance_tau_fixed_v" in sp.columns
                else 0.30
            )
            tau_txt = f" (|V|>{tau_v:.2f} V)"
        e = sp.pivot(index="frequency_khz", columns="scenario_name", values=ex_col).sort_index()
        ex_col_ms = (
            ex_col if (ms_sp is not None and ex_col in ms_sp.columns) else "exceedance_probability"
        )
        if ms_sp is not None and ex_col_ms in ms_sp.columns:
            e_med = (
                ms_sp.groupby(["frequency_khz", "scenario_name"])[ex_col_ms]
                .median()
                .reset_index()
                .pivot(index="frequency_khz", columns="scenario_name", values=ex_col_ms)
                .sort_index()
            )
            if not e_med.empty:
                e = e_med
        for c in e.columns:
            c_str = str(c)
            ax_c.plot(
                e.index, e[c], lw=2.6, color=palette.get(c_str, None), label=_scenario_pretty(c_str)
            )
            if ms_sp is not None and ex_col_ms in ms_sp.columns:
                ms_c = ms_sp[ms_sp["scenario_name"].astype(str) == c_str]
                if not ms_c.empty:
                    q = (
                        ms_c.groupby("frequency_khz")[ex_col_ms]
                        .quantile([0.025, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.975: "q975"})
                        .reset_index()
                        .sort_values("frequency_khz")
                    )
                    if not q.empty:
                        ax_c.fill_between(
                            q["frequency_khz"].to_numpy(dtype=float),
                            q["q025"].to_numpy(dtype=float),
                            q["q975"].to_numpy(dtype=float),
                            color=palette.get(c_str, None),
                            alpha=0.12,
                            linewidth=0.0,
                        )
                        has_spatial_bands = True
        ax_b.set_title("P99 vs frequency", fontsize=13, fontweight="semibold", pad=8)
        ax_b.set_ylabel("P99(|V|) [V]", fontsize=13, labelpad=6)
        ax_b.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax_b.tick_params(labelsize=12)
        ax_b.legend(frameon=False, fontsize=11)
        if has_spatial_bands:
            nseeds = spatial_band_seed_count if spatial_band_seed_count is not None else 5
            ax_b.text(
                0.02,
                0.95,
                f"Shaded bands: 95% across {nseeds} seeds",
                transform=ax_b.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        style(ax_b)
        ax_b.grid(alpha=0.30, linewidth=1.0)
        _panel_label(ax_b, "(b)")

        ax_c.set_title("Exceedance vs frequency", fontsize=13, fontweight="semibold", pad=8)
        ax_c.set_ylabel(f"Exceedance probability{tau_txt}", fontsize=13, labelpad=8)
        ax_c.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax_c.tick_params(labelsize=12)
        if ex_col == "exceedance_probability_fixed_tau03":
            ax_c.text(
                0.02,
                0.95,
                "Fixed screening threshold",
                transform=ax_c.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        style(ax_c)
        ax_c.grid(alpha=0.30, linewidth=1.0)
        _panel_label(ax_c, "(c)")

        g["figure_4_6_composite"] = save_fig(fig, out, "figure_4_6_composite.png")

    if i.coherence is not None:
        c = i.coherence
        f = min(sorted(c["frequency_khz"].unique().tolist()), key=lambda x: abs(x - 30.0))
        sub = c[np.isclose(c["frequency_khz"], f)].sort_values("coherence")
        ex_col = (
            "exceedance_probability_fixed_tau03"
            if "exceedance_probability_fixed_tau03" in sub.columns
            else "exceedance_probability"
        )
        tau_txt = ""
        if ex_col == "exceedance_probability_fixed_tau03":
            tau_v = (
                float(sub["exceedance_tau_fixed_v"].iloc[0])
                if "exceedance_tau_fixed_v" in sub.columns
                else 0.30
            )
            tau_txt = f" (|V|>{tau_v:.2f} V)"
        fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
        ax2 = ax1.twinx()
        has_coh_bands = False
        coh_band_seed_count = None
        coh_x = sub["coherence"].to_numpy(dtype=float)
        coh_p99 = sub["p99_abs_v"].to_numpy(dtype=float)
        coh_ex = sub[ex_col].to_numpy(dtype=float)
        if i.coherence_multiseed is not None and {
            "frequency_khz",
            "coherence",
            "seed",
            "p99_abs_v",
        }.issubset(i.coherence_multiseed.columns):
            ms = i.coherence_multiseed[np.isclose(i.coherence_multiseed["frequency_khz"], f)].copy()
            if not ms.empty:
                coh_band_seed_count = int(ms["seed"].nunique())
                q_p99 = (
                    ms.groupby("coherence")["p99_abs_v"]
                    .quantile([0.025, 0.5, 0.975])
                    .unstack()
                    .rename(columns={0.025: "q025", 0.5: "q500", 0.975: "q975"})
                    .reset_index()
                    .sort_values("coherence")
                )
                if not q_p99.empty:
                    coh_x = q_p99["coherence"].to_numpy(dtype=float)
                    coh_p99 = q_p99["q500"].to_numpy(dtype=float)
                    ax1.fill_between(
                        q_p99["coherence"].to_numpy(dtype=float),
                        q_p99["q025"].to_numpy(dtype=float),
                        q_p99["q975"].to_numpy(dtype=float),
                        color="#0072B2",
                        alpha=0.16,
                        linewidth=0.0,
                    )
                    has_coh_bands = True
                ex_col_ms = ex_col if ex_col in ms.columns else "exceedance_probability"
                if ex_col_ms in ms.columns:
                    q_ex = (
                        ms.groupby("coherence")[ex_col_ms]
                        .quantile([0.025, 0.5, 0.975])
                        .unstack()
                        .rename(columns={0.025: "q025", 0.5: "q500", 0.975: "q975"})
                        .reset_index()
                        .sort_values("coherence")
                    )
                    if not q_ex.empty:
                        coh_ex = q_ex["q500"].to_numpy(dtype=float)
                        ax2.fill_between(
                            q_ex["coherence"].to_numpy(dtype=float),
                            q_ex["q025"].to_numpy(dtype=float),
                            q_ex["q975"].to_numpy(dtype=float),
                            color="#D55E00",
                            alpha=0.14,
                            linewidth=0.0,
                        )
                        has_coh_bands = True
        ax1.plot(
            coh_x,
            coh_p99,
            marker="o",
            lw=2.8,
            color="#0072B2",
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        ax2.plot(
            coh_x,
            coh_ex,
            marker="s",
            lw=2.4,
            linestyle="--",
            color="#D55E00",
            markersize=6.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        ax1.set_title(
            f"Figure 5: Coherence Sweep at {f:.0f} kHz", fontsize=14, fontweight="semibold", pad=10
        )
        ax1.set_xlabel("Coherence parameter", fontsize=13, labelpad=6)
        ax1.set_ylabel("P99(|V|) [V]", fontsize=13, labelpad=6)
        ax2.set_ylabel(f"Exceedance probability{tau_txt}", fontsize=13, labelpad=8)
        ax1.tick_params(labelsize=12)
        ax2.tick_params(labelsize=12)
        if has_coh_bands:
            nseeds = coh_band_seed_count if coh_band_seed_count is not None else 5
            ax1.text(
                0.02,
                0.95,
                f"Shaded bands: 95% across {nseeds} seeds",
                transform=ax1.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        style(ax1)
        ax1.grid(alpha=0.30, linewidth=1.0)
        # Keep Figure 5 clean; dataset/method notes belong in manuscript caption text.
        g["figure_5"] = save_fig(fig, out, "figure_5_phase_coherence_sweep.png")

    if i.feeder_detail is not None:
        d = i.feeder_detail
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
        fig._skip_tight_layout = True
        # Reserve a clean top band for a shared legend (avoid overlaps and "floating" bottom legends).
        fig.subplots_adjust(top=0.78, bottom=0.20, right=0.98, wspace=0.26)
        feeder_palette = {
            "A": "#0072B2",
            "B": "#E69F00",
            "C": "#009E73",
        }
        for fid, sub in d.groupby("feeder_id"):
            fid_s = str(fid)
            color = feeder_palette.get(fid_s, None)
            axes[0].scatter(
                sub["pred_rms_abs_v"],
                sub["sim_rms_abs_v"],
                s=28,
                alpha=0.80,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                marker="o",
                label=f"F{fid_s}",
            )
            axes[1].scatter(
                sub["pred_p99_abs_v"],
                sub["sim_p99_abs_v"],
                s=28,
                alpha=0.80,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                marker="o",
                label=None,
            )
        panel_specs = [
            (
                axes[0],
                "pred_rms_abs_v",
                "sim_rms_abs_v",
                "RMS",
                "Predicted RMS [V]",
                "Simulated RMS [V]",
            ),
            (
                axes[1],
                "pred_p99_abs_v",
                "sim_p99_abs_v",
                "P99",
                "Predicted P99 [V]",
                "Simulated P99 [V]",
            ),
        ]
        for ax, x, y, panel_title, x_label, y_label in panel_specs:
            valid = d[[x, y]].dropna()
            lo = float(min(valid[x].min(), valid[y].min()))
            hi = float(max(valid[x].max(), valid[y].max()))
            span = max(hi - lo, 1e-9)
            pad = 0.02 * span
            lo_lim = lo - pad
            hi_lim = hi + pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.8, alpha=0.85)
            err = np.abs(valid[y].to_numpy(dtype=float) - valid[x].to_numpy(dtype=float))
            mae = float(np.mean(err))
            rmse = float(np.sqrt(np.mean(err**2)))
            ax.text(
                0.03,
                0.95,
                f"MAE={mae:.3f} V\nRMSE={rmse:.3f} V\nn={len(valid)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
            style(ax)
            ax.grid(False)
            ax.set_title(panel_title, fontsize=13, fontweight="semibold", pad=8)
            ax.set_xlabel(x_label, fontsize=13, labelpad=6)
            ax.set_ylabel(y_label, fontsize=13, labelpad=6)
            ax.tick_params(labelsize=12)
            ax.set_xlim(lo_lim, hi_lim)
            ax.set_ylim(lo_lim, hi_lim)
            ax.set_aspect("equal", adjustable="box")

            feeder_lines: list[str] = []
            for fid in sorted(d["feeder_id"].dropna().unique().tolist()):
                sub_f = d[d["feeder_id"] == fid][[x, y]].dropna()
                if sub_f.empty:
                    continue
                err_f = np.abs(sub_f[y].to_numpy(dtype=float) - sub_f[x].to_numpy(dtype=float))
                mae_f = float(np.mean(err_f))
                rmse_f = float(np.sqrt(np.mean(err_f**2)))
                feeder_lines.append(f"F{fid}: MAE {mae_f:.3f}, RMSE {rmse_f:.3f}")
            if feeder_lines:
                ax.text(
                    0.97,
                    0.03,
                    "\n".join(feeder_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.90, edgecolor="none"),
                )
        # One shared legend to keep both panels consistent.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                frameon=False,
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                fontsize=12,
                markerscale=1.2,
                handletextpad=0.6,
                columnspacing=2.6,
            )
        fig.suptitle(
            "Figure 7: Feeder Synthetic Split-Consistency Scatter",
            fontsize=14,
            fontweight="semibold",
            y=0.975,
        )
        g["figure_7"] = save_fig(fig, out, "figure_7_feeder_validation_scatter.png")

    if i.density is not None:
        d = i.density
        # Use a fixed compatibility-style threshold for screening (avoid circular thresholding from the same sweep).
        thr = 0.30
        density_vals = np.sort(d["density"].to_numpy(dtype=float))
        d_min = float(np.min(density_vals))
        d_max = float(np.max(density_vals))
        rows = []
        for f, sub in d.groupby("frequency_khz"):
            s = sub.sort_values("density")
            dens = s["density"].to_numpy(dtype=float)
            p99 = s["p99_abs_v"].to_numpy(dtype=float)
            feasible = p99 <= thr
            if np.any(feasible):
                idx_hi = int(np.where(feasible)[0][-1])
                if idx_hi >= len(dens) - 1:
                    allow = float(dens[idx_hi])
                else:
                    d_lo = float(dens[idx_hi])
                    d_up = float(dens[idx_hi + 1])
                    p_lo = float(p99[idx_hi])
                    p_up = float(p99[idx_hi + 1])
                    if abs(p_up - p_lo) <= 1e-12:
                        allow = d_lo
                    else:
                        frac = (thr - p_lo) / (p_up - p_lo)
                        frac = float(np.clip(frac, 0.0, 1.0))
                        allow = d_lo + frac * (d_up - d_lo)
                rows.append(
                    {
                        "frequency_khz": float(f),
                        "allowable_density": allow,
                        "fallback_density": np.nan,
                    }
                )
            else:
                best_idx = int(np.argmin(p99))
                rows.append(
                    {
                        "frequency_khz": float(f),
                        "allowable_density": np.nan,
                        "fallback_density": float(dens[best_idx]),
                    }
                )
        ad = pd.DataFrame(rows).sort_values("frequency_khz")
        if ad["allowable_density"].notna().any():
            ad["allowable_density_smooth"] = (
                ad["allowable_density"].rolling(window=7, center=True, min_periods=1).median()
            )
        else:
            ad["allowable_density_smooth"] = np.nan
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        any_feasible = bool(ad["allowable_density"].notna().any())
        if ad["allowable_density"].notna().any():
            ax.plot(
                ad["frequency_khz"],
                ad["allowable_density_smooth"],
                lw=2.8,
                marker="o",
                markersize=6.5,
                color="#0072B2",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markevery=6,
                label=rf"allowable density frontier (P99 ≤ {thr:.2f} V)",
            )
        if ad["fallback_density"].notna().any():
            ax.plot(
                ad["frequency_khz"],
                ad["fallback_density"],
                lw=2.2,
                marker="x",
                linestyle="--",
                color="#333333",
                label="lowest tested density (no feasible point)",
            )
        ax.set_title(
            rf"Figure 8: Density Screening at $\tau={thr:.2f}$ V",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        ax.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax.set_ylabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax.tick_params(labelsize=12)
        if not any_feasible:
            ax.text(
                0.02,
                0.96,
                f"No feasible density in tested grid [{d_min:g}, {d_max:g}]",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        ax.legend(frameon=False, fontsize=12)
        style(ax)
        ax.grid(alpha=0.30, linewidth=1.0)
        g["figure_8"] = save_fig(fig, out, "figure_8_allowable_density_screening_chart.png")

        tau_levels = [0.20, 0.30, 0.50]
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        tau_colors = {0.20: "#0072B2", 0.30: "#E69F00", 0.50: "#009E73"}
        for tau in tau_levels:
            rows = []
            for f_hz, sub in d.groupby("frequency_khz"):
                s = sub.sort_values("density")
                dens = s["density"].to_numpy(dtype=float)
                p99 = s["p99_abs_v"].to_numpy(dtype=float)
                feasible = p99 <= tau
                if np.any(feasible):
                    idx_hi = int(np.where(feasible)[0][-1])
                    if idx_hi >= len(dens) - 1:
                        value = float(dens[idx_hi])
                    else:
                        d_lo = float(dens[idx_hi])
                        d_up = float(dens[idx_hi + 1])
                        p_lo = float(p99[idx_hi])
                        p_up = float(p99[idx_hi + 1])
                        if abs(p_up - p_lo) <= 1e-12:
                            value = d_lo
                        else:
                            frac = (tau - p_lo) / (p_up - p_lo)
                            frac = float(np.clip(frac, 0.0, 1.0))
                            value = d_lo + frac * (d_up - d_lo)
                else:
                    value = np.nan
                rows.append({"frequency_khz": float(f_hz), "allowable_density": value})
            curve = pd.DataFrame(rows).sort_values("frequency_khz")
            curve["allowable_density"] = (
                curve["allowable_density"].rolling(window=7, center=True, min_periods=1).median()
            )
            ax.plot(
                curve["frequency_khz"],
                curve["allowable_density"],
                marker="o",
                markersize=5.2,
                markevery=6,
                lw=2.6,
                color=tau_colors.get(float(tau), None),
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=rf"$\tau={tau:.2f}$ V",
            )
        ax.set_title(
            "Figure 8b: Allowable Density Sensitivity to Threshold",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        ax.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax.set_ylabel("Allowable density", fontsize=13, labelpad=6)
        ax.tick_params(labelsize=12)
        ax.legend(frameon=False, fontsize=12, loc="lower right", bbox_to_anchor=(0.98, 0.12))
        style(ax)
        ax.grid(False)
        g["figure_8b"] = save_fig(fig, out, "figure_8b_allowable_density_threshold_sensitivity.png")

        # Composite (Word): Figure 8 + Figure 8b (keep originals as well).
        fig = plt.figure(figsize=(10.8, 4.6))
        fig._skip_tight_layout = True
        gs = fig.add_gridspec(1, 2, wspace=0.22)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])

        # (a) Density screening at fixed threshold
        thr = 0.30
        density_vals = np.sort(d["density"].to_numpy(dtype=float))
        d_min = float(np.min(density_vals))
        d_max = float(np.max(density_vals))
        rows = []
        for f, sub in d.groupby("frequency_khz"):
            s = sub.sort_values("density")
            dens = s["density"].to_numpy(dtype=float)
            p99 = s["p99_abs_v"].to_numpy(dtype=float)
            feasible = p99 <= thr
            if np.any(feasible):
                idx_hi = int(np.where(feasible)[0][-1])
                if idx_hi >= len(dens) - 1:
                    allow = float(dens[idx_hi])
                else:
                    d_lo = float(dens[idx_hi])
                    d_up = float(dens[idx_hi + 1])
                    p_lo = float(p99[idx_hi])
                    p_up = float(p99[idx_hi + 1])
                    if abs(p_up - p_lo) <= 1e-12:
                        allow = d_lo
                    else:
                        frac = (thr - p_lo) / (p_up - p_lo)
                        frac = float(np.clip(frac, 0.0, 1.0))
                        allow = d_lo + frac * (d_up - d_lo)
                rows.append(
                    {
                        "frequency_khz": float(f),
                        "allowable_density": allow,
                        "fallback_density": np.nan,
                    }
                )
            else:
                best_idx = int(np.argmin(p99))
                rows.append(
                    {
                        "frequency_khz": float(f),
                        "allowable_density": np.nan,
                        "fallback_density": float(dens[best_idx]),
                    }
                )
        ad = pd.DataFrame(rows).sort_values("frequency_khz")
        if ad["allowable_density"].notna().any():
            ad["allowable_density_smooth"] = (
                ad["allowable_density"].rolling(window=7, center=True, min_periods=1).median()
            )
        else:
            ad["allowable_density_smooth"] = np.nan
        any_feasible = bool(ad["allowable_density"].notna().any())
        if ad["allowable_density"].notna().any():
            ax_a.plot(
                ad["frequency_khz"],
                ad["allowable_density_smooth"],
                lw=2.8,
                marker="o",
                markersize=6.0,
                markevery=6,
                color="#0072B2",
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=rf"allowable density frontier (P99 ≤ {thr:.2f} V)",
            )
        if ad["fallback_density"].notna().any():
            ax_a.plot(
                ad["frequency_khz"],
                ad["fallback_density"],
                lw=2.2,
                marker="x",
                linestyle="--",
                color="#333333",
                label="lowest tested density (no feasible point)",
            )
        ax_a.set_title(
            rf"Density screening ($\tau={thr:.2f}$ V)", fontsize=13, fontweight="semibold", pad=8
        )
        ax_a.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax_a.set_ylabel("Density [scenario units]", fontsize=13, labelpad=6)
        ax_a.tick_params(labelsize=12)
        if not any_feasible:
            ax_a.text(
                0.02,
                0.96,
                f"No feasible density in tested grid [{d_min:g}, {d_max:g}]",
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.92, edgecolor="none"),
            )
        ax_a.legend(frameon=False, fontsize=11, loc="lower right")
        style(ax_a)
        ax_a.grid(alpha=0.30, linewidth=1.0)
        _panel_label(ax_a, "(a)")

        # (b) Allowable-density sensitivity across thresholds
        tau_levels = [0.20, 0.30, 0.50]
        tau_colors = {0.20: "#0072B2", 0.30: "#E69F00", 0.50: "#009E73"}
        for tau in tau_levels:
            rows = []
            for f_hz, sub in d.groupby("frequency_khz"):
                s = sub.sort_values("density")
                dens = s["density"].to_numpy(dtype=float)
                p99 = s["p99_abs_v"].to_numpy(dtype=float)
                feasible = p99 <= tau
                if np.any(feasible):
                    idx_hi = int(np.where(feasible)[0][-1])
                    if idx_hi >= len(dens) - 1:
                        value = float(dens[idx_hi])
                    else:
                        d_lo = float(dens[idx_hi])
                        d_up = float(dens[idx_hi + 1])
                        p_lo = float(p99[idx_hi])
                        p_up = float(p99[idx_hi + 1])
                        if abs(p_up - p_lo) <= 1e-12:
                            value = d_lo
                        else:
                            frac = (tau - p_lo) / (p_up - p_lo)
                            frac = float(np.clip(frac, 0.0, 1.0))
                            value = d_lo + frac * (d_up - d_lo)
                else:
                    value = np.nan
                rows.append({"frequency_khz": float(f_hz), "allowable_density": value})
            curve = pd.DataFrame(rows).sort_values("frequency_khz")
            curve["allowable_density"] = (
                curve["allowable_density"].rolling(window=7, center=True, min_periods=1).median()
            )
            ax_b.plot(
                curve["frequency_khz"],
                curve["allowable_density"],
                marker="o",
                markersize=5.2,
                markevery=6,
                lw=2.6,
                color=tau_colors.get(float(tau), None),
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=rf"$\tau={tau:.2f}$ V",
            )
        ax_b.set_title("Sensitivity to threshold", fontsize=13, fontweight="semibold", pad=8)
        ax_b.set_xlabel("Frequency [kHz]", fontsize=13, labelpad=6)
        ax_b.set_ylabel("Allowable density", fontsize=13, labelpad=6)
        ax_b.tick_params(labelsize=12)
        ax_b.legend(frameon=False, fontsize=11, loc="lower right", bbox_to_anchor=(0.98, 0.12))
        style(ax_b)
        ax_b.grid(False)
        _panel_label(ax_b, "(b)")

        g["figure_8_8b_composite"] = save_fig(fig, out, "figure_8_8b_composite.png")

    appendix_convergence = _make_appendix_monte_carlo_convergence_figure(i, out)
    if appendix_convergence is not None:
        g["figure_s1_monte_carlo_convergence_rms_p99"] = appendix_convergence

    return g


def make_coverage(out: Path, gen: dict[str, str], i: Inputs) -> str:
    has_feeder = (
        i.feeder_specs is not None and i.feeder_kernels is not None and i.feeder_detail is not None
    )
    rows = [
        {"item": "Table 1", "status": "generated", "file": gen.get("table_1", "")},
        {"item": "Table 2", "status": "generated", "file": gen.get("table_2", "")},
        {"item": "Table 3", "status": "generated", "file": gen.get("table_3", "")},
        {"item": "Table 4", "status": "generated", "file": gen.get("table_4", "")},
        {
            "item": "Table 5",
            "status": "generated" if has_feeder else "unavailable",
            "file": gen.get("table_5", ""),
        },
        {
            "item": "Table 6",
            "status": "generated" if has_feeder else "partial",
            "file": gen.get("table_6", ""),
        },
        {
            "item": "Figure 1",
            "status": "generated" if "figure_1" in gen else "unavailable",
            "file": gen.get("figure_1", ""),
        },
        {
            "item": "Figure 2",
            "status": "generated" if i.density is not None else "partial",
            "file": gen.get("figure_2", ""),
        },
        {
            "item": "Figure 2b",
            "status": "generated" if i.density is not None else "partial",
            "file": gen.get("figure_2b", ""),
        },
        {
            "item": "Figure 3",
            "status": "generated" if i.density is not None else "partial",
            "file": gen.get("figure_3", ""),
        },
        {
            "item": "Figure 4",
            "status": "generated" if i.spatial is not None else "partial",
            "file": gen.get("figure_4", ""),
        },
        {
            "item": "Figure 5",
            "status": "generated" if i.coherence is not None else "partial",
            "file": gen.get("figure_5", ""),
        },
        {
            "item": "Figure 6",
            "status": "generated" if i.spatial is not None else "partial",
            "file": gen.get("figure_6", ""),
        },
        {
            "item": "Figure 7",
            "status": "generated" if i.feeder_detail is not None else "partial",
            "file": gen.get("figure_7", ""),
        },
        {
            "item": "Figure 8",
            "status": "generated" if i.density is not None else "partial",
            "file": gen.get("figure_8", ""),
        },
        {
            "item": "Figure 8b",
            "status": "generated" if i.density is not None else "partial",
            "file": gen.get("figure_8b", ""),
        },
        {
            "item": "Appendix Figure S1",
            "status": "generated"
            if "figure_s1_monte_carlo_convergence_rms_p99" in gen
            else "unavailable",
            "file": gen.get("figure_s1_monte_carlo_convergence_rms_p99", ""),
        },
        {
            "item": "Table G1",
            "status": "generated" if "table_g1" in gen else "unavailable",
            "file": gen.get("table_g1", ""),
        },
        {
            "item": "Figure G1",
            "status": "generated" if "figure_g1" in gen else "unavailable",
            "file": gen.get("figure_g1", ""),
        },
        {
            "item": "Figure G2",
            "status": "generated" if "figure_g2" in gen else "unavailable",
            "file": gen.get("figure_g2", ""),
        },
    ]
    return save_csv(pd.DataFrame(rows), out, "outline_coverage_report.csv")


def run(
    syn: Path, bench: Path, out: Path, clear: bool, *, figures_only: bool = False
) -> dict[str, str]:
    if clear:
        clear_out(
            out,
            patterns=("*.png", "*.pdf", "*.json")
            if figures_only
            else ("*.csv", "*.png", "*.pdf", "*.json"),
        )
    out.mkdir(parents=True, exist_ok=True)
    i = load_inputs(syn, bench, require_bench=not figures_only)
    gen = {}
    if not figures_only:
        gen.update(make_tables(i, out))
    gen.update(make_figures(i, out))
    if not figures_only:
        gen["coverage"] = make_coverage(out, gen, i)
    asset_rows = [
        {
            "asset_id": "Table 1",
            "file": gen.get("table_1", ""),
            "status": "generated" if not figures_only else "skipped",
        },
        {
            "asset_id": "Table 2",
            "file": gen.get("table_2", ""),
            "status": "generated" if not figures_only else "skipped",
        },
        {
            "asset_id": "Table 3",
            "file": gen.get("table_3", ""),
            "status": "generated" if not figures_only else "skipped",
        },
        {
            "asset_id": "Table 4",
            "file": gen.get("table_4", ""),
            "status": "generated" if not figures_only else "skipped",
        },
        {
            "asset_id": "Table 5",
            "file": gen.get("table_5", ""),
            "status": "generated"
            if (not figures_only and i.feeder_specs is not None and i.feeder_kernels is not None)
            else ("unavailable" if not figures_only else "skipped"),
        },
        {
            "asset_id": "Table 6",
            "file": gen.get("table_6", ""),
            "status": "generated"
            if (not figures_only and i.feeder_detail is not None)
            else ("partial" if not figures_only else "skipped"),
        },
        {
            "asset_id": "Figure 1",
            "file": gen.get("figure_1", ""),
            "status": "generated" if "figure_1" in gen else "unavailable",
        },
        {
            "asset_id": "Figure 2",
            "file": gen.get("figure_2", ""),
            "status": "generated" if i.density is not None else "partial",
        },
        {
            "asset_id": "Figure 2b",
            "file": gen.get("figure_2b", ""),
            "status": "generated" if i.density is not None else "partial",
        },
        {
            "asset_id": "Figure 3",
            "file": gen.get("figure_3", ""),
            "status": "generated" if i.density is not None else "partial",
        },
        {
            "asset_id": "Figure 4",
            "file": gen.get("figure_4", ""),
            "status": "generated" if i.spatial is not None else "partial",
        },
        {
            "asset_id": "Figure 5",
            "file": gen.get("figure_5", ""),
            "status": "generated" if i.coherence is not None else "partial",
        },
        {
            "asset_id": "Figure 6",
            "file": gen.get("figure_6", ""),
            "status": "generated" if i.spatial is not None else "partial",
        },
        {
            "asset_id": "Figure 7",
            "file": gen.get("figure_7", ""),
            "status": "generated" if i.feeder_detail is not None else "partial",
        },
        {
            "asset_id": "Figure 8",
            "file": gen.get("figure_8", ""),
            "status": "generated" if i.density is not None else "partial",
        },
        {
            "asset_id": "Figure 8b",
            "file": gen.get("figure_8b", ""),
            "status": "generated" if i.density is not None else "partial",
        },
        {
            "asset_id": "Appendix Figure S1",
            "file": gen.get("figure_s1_monte_carlo_convergence_rms_p99", ""),
            "status": "generated"
            if "figure_s1_monte_carlo_convergence_rms_p99" in gen
            else "unavailable",
        },
        {
            "asset_id": "Table G1",
            "file": gen.get("table_g1", ""),
            "status": "generated" if "table_g1" in gen else "unavailable",
        },
        {
            "asset_id": "Figure G1",
            "file": gen.get("figure_g1", ""),
            "status": "generated" if "figure_g1" in gen else "unavailable",
        },
        {
            "asset_id": "Figure G2",
            "file": gen.get("figure_g2", ""),
            "status": "generated" if "figure_g2" in gen else "unavailable",
        },
    ]
    if not figures_only:
        gen["asset_map"] = save_csv(pd.DataFrame(asset_rows), out, "publication_asset_map.csv")
    manifest = {
        "output_dir": str(out),
        "generated": gen,
        "strict_no_dummy_policy": "No placeholder numeric artifacts are synthesized.",
    }
    mp = out / "artifact_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gen["manifest"] = str(mp)
    return gen


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-dir", default="synthetic_data")
    p.add_argument("--benchmark-dir", default="benchmark_reports")
    p.add_argument("--output-dir", default="manuscript/artifacts")
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse()
    gen = run(
        Path(a.synthetic_dir),
        Path(a.benchmark_dir),
        Path(a.output_dir),
        clear=not a.no_clear,
        figures_only=a.figures_only,
    )
    print("=== Reanalysis Complete ===")
    for k in sorted(gen):
        print(f"{k}: {gen[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
