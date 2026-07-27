"""Analyse the SFNO IC-perturbation sweep from 01_run.py: per-config
ensemble mean/std gifs and Munich meteograms over the NAE region, plus a
blow-up summary built from the per-step physical-bounds violation
metrics 01_run.py wrote alongside the field data.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import drop_time, lead_time_hours, nearest_point

paths = ProjPaths()
metrics_dir = paths.perturbation_metrics_path
output_dir = paths.perturbation_analysis_path
output_dir.mkdir(parents=True, exist_ok=True)

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755
# Matches 01_run.py's NAE_LON_MIN/MAX/NAE_LAT_MIN/MAX exactly.
NAE_EXTENT = [-80.0, 40.0, 30.0, 90.0]  # lon_min, lon_max, lat_min, lat_max

CONFIG_NAMES = ["zero", "brown_0.05", "brown_0.01", "brown_0.002", "gaussian_0.05"]

COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"
COLOR_CONTROL = "#C9622A"  # member 0: always Zero() regardless of config - see 01_run.py

GIF_FRAME_DURATION = 0.3
GIF_DPI = 100


def load_config(name):
    zarr_path = paths.perturbation_zarr_path(name)
    if not zarr_path.exists():
        return None
    ds = xr.open_zarr(zarr_path)
    bounds_path = metrics_dir / f"{name}_bounds.csv"
    bounds_df = pd.read_csv(bounds_path) if bounds_path.exists() else pd.DataFrame()
    return ds, bounds_df


def wind_speed(ds):
    return np.hypot(ds["u10m"], ds["v10m"])


FIELD_SPECS = [
    ("t2m", "2m temperature (K)", lambda ds: ds["t2m"], "RdYlBu_r"),
    ("wind_speed10m", "10m wind speed (m/s)", wind_speed, "viridis"),
    ("z500", "500 hPa geopotential (m^2/s^2)", lambda ds: ds["z500"], "RdYlBu_r"),
]


def _robust_range(arr):
    """1st/99th percentile of a possibly-NaN/Inf-containing array - used
    for gif color scales instead of raw min/max, since some configs in
    this sweep are *expected* to genuinely diverge (that's the point of
    the sweep) and a single blown-up member/step would otherwise make the
    whole color scale useless for every other frame. Falls back to (0, 1)
    if nothing finite remains."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    return float(lo), float(hi) if hi > lo else float(lo) + 1e-6


# ---------------------------------------------------------------------------
# Per-config mean/std gif: 3 fields (t2m/wind10m/z500) x 2 (mean, std)
# ---------------------------------------------------------------------------
def render_mean_std_gif(ds, config_name, out_path):
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lon, lat = ds[lon_name].values, ds[lat_name].values
    n_steps = ds.sizes["lead_time"]

    panels = []
    for key, label, fn, cmap in FIELD_SPECS:
        arr = drop_time(fn(ds)).compute()
        mean = arr.mean(dim="ensemble")
        std = arr.std(dim="ensemble")
        mean_lo, mean_hi = _robust_range(mean.values)
        std_lo, std_hi = 0.0, max(_robust_range(std.values)[1], 1e-6)
        panels.append((label, cmap, mean, std, mean_lo, mean_hi, std_lo, std_hi))

    fig, axes = plt.subplots(
        len(FIELD_SPECS), 2, figsize=(10, 3.6 * len(FIELD_SPECS)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    meshes = []
    for row, (label, cmap, mean, std, mean_lo, mean_hi, std_lo, std_hi) in enumerate(panels):
        for col, (arr, clo, chi, cm, title) in enumerate([
            (mean, mean_lo, mean_hi, cmap, f"{label} - ensemble mean"),
            (std, std_lo, std_hi, "viridis", f"{label} - ensemble std"),
        ]):
            ax = axes[row, col]
            ax.set_extent(NAE_EXTENT, crs=ccrs.PlateCarree())
            ax.coastlines(resolution="50m", linewidth=0.6, color="#444444")
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#444444")
            data0 = np.nan_to_num(arr.isel(lead_time=0).values, nan=clo)
            mesh = ax.pcolormesh(lon, lat, data0, transform=ccrs.PlateCarree(), cmap=cm, vmin=clo, vmax=chi, shading="auto")
            fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
            ax.set_title(title, fontsize=9)
            meshes.append((mesh, arr, clo))

    frames = []
    for step in range(n_steps):
        for mesh, arr, clo in meshes:
            data = np.nan_to_num(arr.isel(lead_time=step).values, nan=clo)
            mesh.set_array(data.ravel())
        fig.suptitle(f"{config_name} - lead time step {step}", fontsize=11)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION, loop=0)


# ---------------------------------------------------------------------------
# Munich meteograms: t2m and wind10m speed, member 0 (control) highlighted
# ---------------------------------------------------------------------------
def plot_munich_meteograms(ds, config_name, out_dir):
    munich = nearest_point(ds, MUNICH_LAT, MUNICH_LON)
    hours = lead_time_hours(ds)

    t2m = drop_time(munich["t2m"]).transpose("lead_time", "ensemble").compute().values - 273.15
    wind = drop_time(wind_speed(munich)).transpose("lead_time", "ensemble").compute().values

    for key, data_2d, ylabel, title in [
        ("t2m", t2m, "Temperature (deg C)", f"{config_name} - Munich 2m temperature"),
        ("wind10m", wind, "Wind speed (m/s)", f"{config_name} - Munich 10m wind speed"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.boxplot(
            list(data_2d), positions=np.arange(data_2d.shape[0]), widths=0.6, patch_artist=True, showfliers=False,
            medianprops=dict(color=COLOR_MEAN, linewidth=1.5),
            boxprops=dict(facecolor=COLOR_MEMBER, alpha=0.5, edgecolor="#444444", linewidth=0.6),
            whiskerprops=dict(color="#444444", linewidth=0.6), capprops=dict(color="#444444", linewidth=0.6),
        )
        # member 0 is always the Zero() control (see 01_run.py) - overlaid
        # explicitly since it would otherwise be indistinguishable inside
        # the member-agnostic boxplot aggregate.
        ax.plot(np.arange(data_2d.shape[0]), data_2d[:, 0], color=COLOR_CONTROL,
                 marker="o", markersize=3, linewidth=1.2, label="control (member 0, Zero())")
        step = max(1, data_2d.shape[0] // 20)
        ax.set_xticks(np.arange(0, data_2d.shape[0], step))
        ax.set_xticklabels([f"{h:.0f}" for h in hours[::step]], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Lead time (hours since UTC init)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"{config_name}_meteogram_{key}_munich.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Blow-up summary: worst (max across all 73 variables) violating_fraction
# per (config, member), over lead_time
# ---------------------------------------------------------------------------
def plot_blowup_summary(all_bounds, out_path):
    """Flat near SFNO's own ~7% tcwv baseline (see 01_run.py's docstring)
    means "no genuine blow-up"; a sharp rise toward 1.0 means the member
    diverged. member 0 (control, always Zero()) should sit at that same
    baseline in every panel - if it doesn't, something upstream broke."""
    names = [n for n in CONFIG_NAMES if n in all_bounds and not all_bounds[n].empty]
    if not names:
        print("[WARN] No bounds data found - skipping blow-up summary.")
        return

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        df = all_bounds[name]
        worst = df.groupby(["member", "step", "lead_time_hours"])["violating_fraction"].max().reset_index()
        for member_id, member_df in worst.groupby("member"):
            member_df = member_df.sort_values("lead_time_hours")
            is_control = member_id == 0
            ax.plot(
                member_df["lead_time_hours"], member_df["violating_fraction"],
                color=COLOR_CONTROL if is_control else COLOR_MEMBER,
                linewidth=1.8 if is_control else 1.2, alpha=1.0 if is_control else 0.7,
                label="control (member 0)" if is_control else None,
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color="#DDDDDD", linewidth=0.6)

    axes[0].set_ylabel("Worst-variable violating_fraction")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Blow-up summary: worst per-variable bounds-violating fraction, per member")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    all_bounds = {}
    for name in CONFIG_NAMES:
        result = load_config(name)
        if result is None:
            print(f"[WARN] '{name}': no data found at {paths.perturbation_zarr_path(name)} - run 01_run.py first. Skipping.")
            continue
        ds, bounds_df = result
        all_bounds[name] = bounds_df

        print(f"Rendering mean/std gif for '{name}'...")
        render_mean_std_gif(ds, name, output_dir / f"{name}_mean_std.gif")

        print(f"Rendering Munich meteograms for '{name}'...")
        plot_munich_meteograms(ds, name, output_dir)

    print("Rendering blow-up summary...")
    plot_blowup_summary(all_bounds, output_dir / "blowup_summary.png")
    print(f"\nDone. Outputs written to {output_dir}/")


if __name__ == "__main__":
    main()
