"""Verify what per-step variable subsetting (output_coords={"variable": ...}
passed to run.ensemble) actually buys you.

pipeline/ensemble/01_run.py already uses this to trim the zarr output to a
handful of variables. Confirmed by reading earth2studio.run.ensemble's
source: prognostic.create_iterator() always yields the model's *full*
internal state every step (FCN3 needs the whole state to roll forward), and
output_coords only filters immediately before io.write(). So subsetting
should cut storage and write time, but NOT model compute time.

This script runs the same small ensemble twice - once with the full variable
set, once with a small subset - and reports wall-clock time and on-disk zarr
size for both, so the claim above is checked empirically rather than assumed
from reading the source.
"""

import csv
import shutil
import time

import numpy as np

from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import ensemble as run_ensemble
from earth2studio.perturbation import Zero

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

N_ENSEMBLE = 2
N_STEPS = 4
START_DATE = "2026-07-23T00:00:00"  # UTC - GFS timestamps are always UTC
N_BATCH_SIZE = 2

SUBSET_VARIABLES = ["u10m", "v10m", "t2m", "z500"]


def dir_size_bytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_once(model, data, zarr_path, output_coords):
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    io = ZarrBackend(str(zarr_path))
    perturbation = Zero()

    start = time.perf_counter()
    run_ensemble(
        time=[START_DATE],
        nsteps=N_STEPS,
        nensemble=N_ENSEMBLE,
        prognostic=model,
        data=data,
        io=io,
        perturbation=perturbation,
        batch_size=N_BATCH_SIZE,
        output_coords=output_coords,
    )
    elapsed = time.perf_counter() - start
    size = dir_size_bytes(zarr_path)
    return elapsed, size


def main():
    print("Loading FCN3 model weights...")
    package = FCN3.load_default_package()
    model = FCN3.load_model(package)
    full_variables = list(model.output_coords(model.input_coords())["variable"])
    data = GFS()

    out_dir = paths.variable_subset_data_path
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full-variable-set ({len(full_variables)} vars) forecast...")
    full_time, full_size = run_once(
        model, data, out_dir / "full.zarr", output_coords={}
    )

    print(f"Running subset ({len(SUBSET_VARIABLES)} vars) forecast...")
    subset_time, subset_size = run_once(
        model, data, out_dir / "subset.zarr",
        output_coords={"variable": np.array(SUBSET_VARIABLES)},
    )

    paths.variable_subset_output_path.mkdir(parents=True, exist_ok=True)

    csv_path = paths.variable_subset_output_path / "full_vs_subset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "n_variables", "wall_time_s", "size_mb"])
        writer.writeheader()
        writer.writerow({"run": "full", "n_variables": len(full_variables), "wall_time_s": full_time, "size_mb": full_size / 1e6})
        writer.writerow({"run": "subset", "n_variables": len(SUBSET_VARIABLES), "wall_time_s": subset_time, "size_mb": subset_size / 1e6})
    print(f"CSV written to {csv_path}")

    report_path = paths.variable_subset_output_path / "full_vs_subset.md"
    report = f"""# Variable subset write comparison

nensemble={N_ENSEMBLE}, nsteps={N_STEPS}

| Run | # variables | Wall time (s) | Zarr size (MB) |
|-----|-------------|----------------|-----------------|
| Full | {len(full_variables)} | {full_time:.1f} | {full_size / 1e6:.1f} |
| Subset | {len(SUBSET_VARIABLES)} | {subset_time:.1f} | {subset_size / 1e6:.1f} |

Storage ratio (subset/full): {subset_size / full_size:.3f}
Time ratio (subset/full): {subset_time / full_time:.3f}

Model compute per step is unchanged either way (confirmed by reading
run.ensemble's source - see this script's docstring). If the time ratio
isn't close to 1.0, that means zarr write I/O is a substantial fraction of
total wall time for this run size (nsteps={N_STEPS}, nensemble={N_ENSEMBLE}),
not that the model computed less work - for a longer rollout, where I/O is
a smaller share of the total, expect the ratio to drift closer to 1.0.
"""
    report_path.write_text(report)
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
