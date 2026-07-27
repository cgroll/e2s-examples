"""Analyse the SFNO IC-perturbation sweep from 01_run.py: per-config
ensemble mean/std gifs, Munich meteograms, and a diagnostics dashboard
built from the per-step bounds, cross-variable, and global-mean-timeseries
metrics 01_run.py wrote alongside the field data.

The dashboard is deliberately one full-size figure per config (five
stacked panels), not five configs squeezed into one row of subplots -
each panel needs enough width to actually read, and cramming all five
configs side by side left everything too small to make out.
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
COLOR_VARIABLE_PALETTE = [
    "#1F5C99", "#C9622A", "#2E8B57", "#6A4C93", "#D64545",
    "#E68A2E", "#4C78A8", "#54A24B", "#B279A2", "#9D755D",
]

GIF_FRAME_DURATION = 0.3
GIF_DPI = 100

# Same thresholds pipeline/ensemble/02_validate.py uses for FCN3's
# cross-time checks - reused here as reference lines, not gates (see
# 01_run.py's docstring on why nothing here gates during inference).
KE_GROWTH_FACTOR = 5.0
MASS_DRIFT_TOLERANCE = 0.01


def load_config(name):
    zarr_path = paths.perturbation_zarr_path(name)
    if not zarr_path.exists():
        return None
    ds = xr.open_zarr(zarr_path)

    def read_csv(suffix):
        path = metrics_dir / f"{name}_{suffix}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return ds, read_csv("bounds"), read_csv("cross_variable"), read_csv("timeseries")


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


def _geographic_lon_order(lon):
    """Plot-time-only reordering (plain numpy, never applied to the
    source Dataset/its coordinate index) of a lon array that's
    numerically ascending but geographically split by the 0/360 seam -
    NAE's stored lon is [0...40, 280...360), ascending (required for
    xarray's .sel(method="nearest"), used by the Munich meteograms below)
    but not geographically contiguous. pcolormesh draws consecutive array
    entries as adjacent, so plotting that stored order draws one giant
    quad spanning the 240 degree gap between index "40" and index "280" -
    the vertical seam bug. Returns (pm180_lon_sorted, sort_order); apply
    sort_order to the data's lon axis before plotting pm180_lon_sorted.
    See 01_run.py's nae_crop_masks docstring for why this is fixed here,
    at plot time, rather than in storage order."""
    lon_pm180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_pm180)
    return lon_pm180[order], order


# ---------------------------------------------------------------------------
# Per-config mean/std gif: 3 fields (t2m/wind10m/z500) x 2 (mean, std)
# ---------------------------------------------------------------------------
def render_mean_std_gif(ds, config_name, out_path):
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lat = ds[lat_name].values
    lon, lon_order = _geographic_lon_order(ds[lon_name].values)
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
            data0 = np.nan_to_num(arr.isel(lead_time=0).values[:, lon_order], nan=clo)
            mesh = ax.pcolormesh(lon, lat, data0, transform=ccrs.PlateCarree(), cmap=cm, vmin=clo, vmax=chi, shading="auto")
            fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
            ax.set_title(title, fontsize=9)
            meshes.append((mesh, arr, clo))

    frames = []
    for step in range(n_steps):
        for mesh, arr, clo in meshes:
            data = np.nan_to_num(arr.isel(lead_time=step).values[:, lon_order], nan=clo)
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
# Per-config diagnostics dashboard - five full-width panels answering:
# (A) are all members affected, and is the onset immediate or gradual?
# (B) are all variables affected, or only some?
# (C) is it just absolute bounds, or does relative ordering break too?
# (D/E) is energy/mass conserved over the rollout?
# ---------------------------------------------------------------------------
def _member_lines(ax, df, value_col, group_cols=("member", "step", "lead_time_hours"), agg="max"):
    """One line per member, control (member 0) highlighted - the shared
    layout for panels A and C below."""
    per_member = df.groupby(list(group_cols))[value_col].agg(agg).reset_index()
    for member_id, member_df in per_member.groupby("member"):
        member_df = member_df.sort_values("lead_time_hours")
        is_control = member_id == 0
        ax.plot(
            member_df["lead_time_hours"], member_df[value_col],
            color=COLOR_CONTROL if is_control else COLOR_MEMBER,
            linewidth=2.0 if is_control else 1.3, alpha=1.0 if is_control else 0.75,
            label="control (member 0)" if is_control else f"member {member_id}",
        )


def plot_config_dashboard(name, bounds_df, cross_variable_df, timeseries_df, out_path):
    if bounds_df.empty:
        print(f"[WARN] '{name}': no bounds data - skipping dashboard.")
        return

    fig, axes = plt.subplots(5, 1, figsize=(14, 22))
    ax_bounds, ax_vars, ax_cross, ax_ke, ax_mass = axes

    # --- A: per-member worst-variable violating_fraction over time -----
    _member_lines(ax_bounds, bounds_df, "violating_fraction")
    ax_bounds.set_title(f"{name}  -  (A) worst-variable bounds-violating fraction, per member")
    ax_bounds.set_ylabel("Violating fraction")
    ax_bounds.set_ylim(-0.02, 1.02)
    ax_bounds.legend(loc="upper left", fontsize=8, ncol=2)

    # --- B: per-variable breakdown, perturbed members only -------------
    perturbed = bounds_df[bounds_df["member"] != 0]
    if not perturbed.empty:
        per_var = perturbed.groupby(["variable", "step", "lead_time_hours"])["violating_fraction"].mean().reset_index()
        final_step = per_var["step"].max()
        top_vars = (
            per_var[per_var["step"] == final_step]
            .sort_values("violating_fraction", ascending=False)["variable"].head(10).tolist()
        )
        for i, var in enumerate(top_vars):
            var_df = per_var[per_var["variable"] == var].sort_values("lead_time_hours")
            ax_vars.plot(
                var_df["lead_time_hours"], var_df["violating_fraction"],
                color=COLOR_VARIABLE_PALETTE[i % len(COLOR_VARIABLE_PALETTE)], linewidth=1.5, label=var,
            )
    ax_vars.set_title("(B) top-10 most-affected variables (mean violating fraction across perturbed members)")
    ax_vars.set_ylabel("Violating fraction")
    ax_vars.set_ylim(-0.02, 1.02)
    ax_vars.legend(loc="upper left", fontsize=7, ncol=5)

    # --- C: cross-variable (z-level ordering) consistency --------------
    if not cross_variable_df.empty:
        _member_lines(ax_cross, cross_variable_df, "violating_fraction")
        ax_cross.legend(loc="upper left", fontsize=8, ncol=2)
    ax_cross.set_title("(C) z-level ordering violations (worst adjacent pair, per member) - not just absolute bounds")
    ax_cross.set_ylabel("Violating fraction")
    ax_cross.set_ylim(-0.02, 1.02)

    # --- D/E: energy and mass conservation vs. step 0 -------------------
    if not timeseries_df.empty and "global_mean_kinetic_energy" in timeseries_df:
        for member_id, member_df in timeseries_df.groupby("member"):
            member_df = member_df.sort_values("lead_time_hours")
            ke0 = member_df["global_mean_kinetic_energy"].iloc[0]
            ratio = member_df["global_mean_kinetic_energy"] / ke0 if ke0 else np.nan
            is_control = member_id == 0
            ax_ke.plot(
                member_df["lead_time_hours"], ratio,
                color=COLOR_CONTROL if is_control else COLOR_MEMBER,
                linewidth=2.0 if is_control else 1.3, alpha=1.0 if is_control else 0.75,
            )
        ax_ke.axhline(KE_GROWTH_FACTOR, color="#D64545", linestyle="--", linewidth=1.0,
                       label=f"02_validate.py's KE_GROWTH_FACTOR ({KE_GROWTH_FACTOR}x)")
        # Log scale: a genuinely blown-up member can reach ratios of 1e4-1e5x,
        # which would otherwise squash the 5x reference line down to
        # indistinguishable-from-zero on a linear axis.
        ax_ke.set_yscale("log")
        ax_ke.legend(loc="upper left", fontsize=8)

        for member_id, member_df in timeseries_df.groupby("member"):
            member_df = member_df.sort_values("lead_time_hours")
            msl0 = member_df["global_mean_msl"].iloc[0]
            drift = (member_df["global_mean_msl"] - msl0) / msl0 if msl0 else np.nan
            is_control = member_id == 0
            ax_mass.plot(
                member_df["lead_time_hours"], drift,
                color=COLOR_CONTROL if is_control else COLOR_MEMBER,
                linewidth=2.0 if is_control else 1.3, alpha=1.0 if is_control else 0.75,
            )
        ax_mass.axhline(MASS_DRIFT_TOLERANCE, color="#D64545", linestyle="--", linewidth=1.0,
                          label=f"02_validate.py's MASS_DRIFT_TOLERANCE (+/-{MASS_DRIFT_TOLERANCE})")
        ax_mass.axhline(-MASS_DRIFT_TOLERANCE, color="#D64545", linestyle="--", linewidth=1.0)
        ax_mass.legend(loc="upper left", fontsize=8)

    ax_ke.set_title("(D) kinetic energy vs. step 0 (ratio) - reference: FCN3's own blow-up threshold")
    ax_ke.set_ylabel("KE ratio")
    ax_mass.set_title("(E) mean sea-level pressure drift vs. step 0 (fractional) - reference: FCN3's own mass-conservation tolerance")
    ax_mass.set_ylabel("Fractional drift")
    ax_mass.set_xlabel("Lead time (hours since UTC init)")

    for ax in axes:
        ax.grid(True, color="#DDDDDD", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for name in CONFIG_NAMES:
        result = load_config(name)
        if result is None:
            print(f"[WARN] '{name}': no data found at {paths.perturbation_zarr_path(name)} - run 01_run.py first. Skipping.")
            continue
        ds, bounds_df, cross_variable_df, timeseries_df = result

        print(f"Rendering mean/std gif for '{name}'...")
        render_mean_std_gif(ds, name, output_dir / f"{name}_mean_std.gif")

        print(f"Rendering Munich meteograms for '{name}'...")
        plot_munich_meteograms(ds, name, output_dir)

        print(f"Rendering diagnostics dashboard for '{name}'...")
        plot_config_dashboard(name, bounds_df, cross_variable_df, timeseries_df, output_dir / f"{name}_dashboard.png")

    print(f"\nDone. Outputs written to {output_dir}/")


if __name__ == "__main__":
    main()
