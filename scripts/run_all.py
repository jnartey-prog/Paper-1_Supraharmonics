"""Entrypoint for full synthetic benchmark re-analysis pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path


# Support direct execution: `python scripts/run_all.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from scripts import config, io, plots, tables
    from scripts import feeder_benchmark, feeder_sim

    t0 = time.time()
    paths = config.build_paths()
    io.ensure_output_dirs(paths)
    io.clear_previous_outputs(paths)
    inputs = io.load_inputs(paths)

    feeder_bench = feeder_benchmark.generate_feeder_benchmark(paths)
    feeder_sim_out = feeder_sim.generate_feeder_model_simulations(
        paths=paths,
        baseline_scenario=inputs["baseline_scenario"],
        kernels_df=feeder_bench["kernels_df"],
    )

    table_outputs = tables.generate_all_tables(inputs, paths)
    figure_outputs = plots.generate_all_figures(inputs, paths)

    elapsed_s = time.time() - t0

    print("=== Synthetic Re-analysis Complete ===")
    print(f"Input root: {paths.root}")
    print(f"Tables dir: {paths.tables_dir}")
    print(f"Figures dir: {paths.figures_dir}")
    print(f"Feeder kernels: {paths.feeder_benchmark_kernels}")
    print(f"Feeder feature table: {paths.feeder_benchmark_features}")
    print(f"Feeder realization file: {paths.feeder_model_realizations}")
    print(f"Feeder summary file: {paths.feeder_model_setting_summary}")
    print("")
    print("Generated tables (file, row_count):")
    for path, rows in table_outputs:
        print(f"- {path} | rows={rows}")
    print("")
    print("Generated figures (file, row_count):")
    for path in figure_outputs:
        print(f"- {path} | rows=n/a")
    print("")
    print(f"Total tables: {len(table_outputs)}")
    print(f"Total figures: {len(figure_outputs)}")
    print(f"Feeder benchmark kernel rows: {int(feeder_bench['kernels_df'].shape[0])}")
    print(f"Feeder benchmark feature rows: {int(feeder_bench['features_df'].shape[0])}")
    print(f"Feeder model realization rows: {int(feeder_sim_out['n_realizations_rows'])}")
    print(f"Feeder model setting rows: {int(feeder_sim_out['summary_df'].shape[0])}")
    print(f"Elapsed seconds: {elapsed_s:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
