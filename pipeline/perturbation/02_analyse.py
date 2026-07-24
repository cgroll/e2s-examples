"""Compare ensemble spread growth across the perturbation strategies run
by 01_run.py: does injecting noise at every rollout step (per_step_brown)
grow spread differently than perturbing only the initial condition -
whether by static noise or by a dynamically-bred (bred-vector) method?
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import area_weights, ensemble_spread_series, lead_time_hours

paths = ProjPaths()
output_dir = paths.perturbation_output_path

CONFIG_LABELS = {
    "zero": "Zero (no perturbation, baseline)",
    "gaussian": "Gaussian (IID, IC-only)",
    "brown": "Brown (correlated, IC-only)",
    "spherical_gaussian": "SphericalGaussian (IC-only)",
    "bred_vector": "BredVector (dynamically grown, IC-only)",
    "hemispheric_bred_vector": "HemisphericCentredBredVector (HENS-style, IC-only)",
    "per_step_brown": "Brown, injected every step (not just IC)",
}

# Visually group by category rather than one arbitrary color per line:
# static IC-only in blue/green tones, bred-vector methods in orange
# tones, the per-step method in a single, distinct red.
CONFIG_COLORS = {
    "zero": "#6E7B8B",
    "gaussian": "#4C78A8",
    "brown": "#1F5C99",
    "spherical_gaussian": "#54A24B",
    "bred_vector": "#E68A2E",
    "hemispheric_bred_vector": "#C9622A",
    "per_step_brown": "#D64545",
}


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    any_plotted = False

    for name, label in CONFIG_LABELS.items():
        zarr_path = paths.perturbation_zarr_path(name)
        if not zarr_path.exists():
            print(f"[WARN] {zarr_path} not found - run 01_run.py first. Skipping '{name}'.")
            continue

        ds = xr.open_zarr(zarr_path)
        if "t2m" not in ds.data_vars or "ensemble" not in ds["t2m"].dims:
            print(f"[WARN] 't2m' missing/no ensemble dim in '{name}', skipping.")
            continue

        hours = lead_time_hours(ds)
        weights = area_weights(ds)
        spread = ensemble_spread_series(ds, weights, "t2m").values
        ax.plot(hours, spread, color=CONFIG_COLORS.get(name, "#333333"), linewidth=2.0, label=label)
        any_plotted = True

    if not any_plotted:
        print("[WARN] No perturbation configs found - nothing to plot. Run 01_run.py first.")
        return

    ax.set_xlabel("Lead time (hours since UTC init)")
    ax.set_ylabel("Ensemble std of global-mean t2m (K)")
    ax.set_title("Ensemble spread growth by perturbation strategy")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    out_path = output_dir / "spread_by_perturbation_strategy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
