import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cartopy.crs as ccrs
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import (
    drop_time,
    global_mean,
    group_and_spatial_dims,
    lead_time_hours,
    load_germany_weights,
    nearest_point,
)

paths = ProjPaths()
zarr_path = paths.ensemble_zarr_path
output_dir = paths.ensemble_analysis_path

# Render every Nth lead_time step into the gifs (1 = every step). Bumping this
# up trades animation smoothness for a lot less render time on long rollouts.
GIF_STEP_STRIDE = 1
GIF_FRAME_DURATION = 0.15  # seconds per frame
GIF_DPI = 100

# Block-average the lat/lon grid by this factor before rendering gifs. The
# output frames are only ~800x450px, well below the native 0.25-deg grid's
# resolution, so this is visually lossless while cutting cartopy's per-mesh
# Robinson reprojection cost (the actual bottleneck) by roughly the same factor.
GIF_SPATIAL_COARSEN_FACTOR = 4

# Fixed color-scale limits so frames/members are visually comparable.
T2M_VMIN, T2M_VCENTER, T2M_VMAX = 220.0, 273.15, 320.0  # Kelvin
WIND_SPEED_MAX = 40.0  # m/s
Z500_VMIN, Z500_VMAX = 45000.0, 59000.0  # m^2/s^2, typical global z500 range

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755

COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"
COLOR_POPWEIGHTED = "#C9622A"
COLOR_DIFF = "#6A4C93"


# ---------------------------------------------------------------------------
# Robinson-projection gifs
# ---------------------------------------------------------------------------
def render_robinson_gif_scalar(ds, member_index, member_id, var_name, out_path, cmap, norm, cbar_label):
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    arr = drop_time(ds[var_name].isel(ensemble=member_index))
    lon, lat = ds[lon_name].values, ds[lat_name].values

    # Figure/axes/mesh are created once and reused across frames: cartopy's
    # pcolormesh setup (antimeridian wrapping over the full global grid) costs
    # ~20s, versus <1s to update an existing mesh's data via set_array. Doing
    # this per-frame instead of per-gif is what made rendering take forever.
    fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="#444444")

    mesh = None
    frames = []
    for step in range(0, arr.sizes["lead_time"], GIF_STEP_STRIDE):
        data = arr.isel(lead_time=step).values
        if mesh is None:
            mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cbar.set_label(cbar_label)
        else:
            mesh.set_array(data.ravel())
        ax.set_title(f"{var_name} - member {member_id} - step {step}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION, loop=0)


def render_robinson_gif_wind(ds, member_index, member_id, out_path):
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    u = drop_time(ds["u10m"].isel(ensemble=member_index))
    v = drop_time(ds["v10m"].isel(ensemble=member_index))
    lon, lat = ds[lon_name].values, ds[lat_name].values

    lon_stride = max(1, len(lon) // 36)
    lat_stride = max(1, len(lat) // 18)

    # See render_robinson_gif_scalar: mesh/quiver are created once and updated
    # in place per frame rather than rebuilt, since cartopy's pcolormesh setup
    # over the full global grid is ~20s vs <1s for a set_array/set_UVC update.
    fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="#444444")

    mesh = None
    quiver = None
    frames = []
    for step in range(0, u.sizes["lead_time"], GIF_STEP_STRIDE):
        u_step = u.isel(lead_time=step).values
        v_step = v.isel(lead_time=step).values
        speed_step = np.hypot(u_step, v_step)

        if mesh is None:
            mesh = ax.pcolormesh(
                lon, lat, speed_step, transform=ccrs.PlateCarree(),
                cmap="viridis", vmin=0, vmax=WIND_SPEED_MAX, shading="auto",
            )
            quiver = ax.quiver(
                lon[::lon_stride], lat[::lat_stride],
                u_step[::lat_stride, ::lon_stride], v_step[::lat_stride, ::lon_stride],
                transform=ccrs.PlateCarree(), color="black", scale=800, width=0.0022,
            )
            cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cbar.set_label("10m wind speed (m/s)")
        else:
            mesh.set_array(speed_step.ravel())
            quiver.set_UVC(u_step[::lat_stride, ::lon_stride], v_step[::lat_stride, ::lon_stride])

        ax.set_title(f"10m wind - member {member_id} - step {step}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION, loop=0)


def render_robinson_gif_ensemble_std(ds, var_name, out_path, cmap, cbar_label):
    """Animate the standard deviation across ensemble members at each grid
    point, over the rollout - a spatial view of ensemble spread, as opposed
    to the per-member scalar gifs above (one member's value) or the Munich
    meteogram (spread at one point only)."""
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    arr = drop_time(ds[var_name]).std(dim="ensemble").compute()
    lon, lat = ds[lon_name].values, ds[lat_name].values
    # Scale from the data itself rather than a fixed constant: unlike t2m/wind
    # value ranges, the spread's magnitude depends on the perturbation method
    # (e.g. Zero() only has model-internal stochasticity) and isn't known a priori.
    vmax = max(float(arr.max()), 1e-6)  # avoid a degenerate 0-width scale if spread is exactly 0

    fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="#444444")

    mesh = None
    frames = []
    for step in range(0, arr.sizes["lead_time"], GIF_STEP_STRIDE):
        data = arr.isel(lead_time=step).values
        if mesh is None:
            mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=vmax, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cbar.set_label(cbar_label)
        else:
            mesh.set_array(data.ravel())
        ax.set_title(f"{var_name} ensemble std - step {step}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION, loop=0)


# ---------------------------------------------------------------------------
# Munich meteograms (ensemble boxplot per lead_time step)
# ---------------------------------------------------------------------------
def coarsen_spatial(ds, factor):
    if factor <= 1:
        return ds
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    return ds.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean()


def plot_meteogram_boxplot(x_hours, data_2d, title, ylabel, out_path):
    """data_2d: shape (n_lead_time, n_ensemble)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        list(data_2d),
        positions=np.arange(data_2d.shape[0]),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=COLOR_MEAN, linewidth=1.5),
        boxprops=dict(facecolor=COLOR_MEMBER, alpha=0.5, edgecolor="#444444", linewidth=0.6),
        whiskerprops=dict(color="#444444", linewidth=0.6),
        capprops=dict(color="#444444", linewidth=0.6),
    )
    step = max(1, data_2d.shape[0] // 20)
    ax.set_xticks(np.arange(0, data_2d.shape[0], step))
    ax.set_xticklabels([f"{h:.0f}" for h in x_hours[::step]], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Lead time (hours since UTC init)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_germany_comparison(x_hours, simple_2d, popweighted_2d, title, ylabel, out_path):
    """simple_2d/popweighted_2d: shape (n_lead_time, n_ensemble) - both
    computed from the same forecast array via global_mean() with
    different weight fields, member-for-member aligned (ensemble index i
    is the same underlying forecast draw in both), so their difference
    below is a genuine per-member difference, not a comparison across
    unrelated samples (see load_germany_weights' docstring).

    Three rows: (1) both means overlaid with no band, so the (usually
    small) gap between weightings is readable without band-on-band
    overlap muddying the color; (2) each weighting on its own axis with
    its own mean + ensemble min-max range, so within-weighting spread
    isn't conflated with between-weighting spread; (3) the per-member
    difference (population-weighted minus area-weighted), mean line plus
    its own range - shows whether that gap is consistent across the
    ensemble or itself uncertain.
    """
    diff_2d = popweighted_2d - simple_2d

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.5, wspace=0.15)
    ax_combined = fig.add_subplot(gs[0, :])
    ax_simple = fig.add_subplot(gs[1, 0])
    ax_pop = fig.add_subplot(gs[1, 1], sharey=ax_simple)
    ax_diff = fig.add_subplot(gs[2, :])

    def band(ax, data_2d, color, label, show_band=True):
        mean = data_2d.mean(axis=1)
        if show_band:
            ax.fill_between(x_hours, data_2d.min(axis=1), data_2d.max(axis=1), color=color, alpha=0.15)
        ax.plot(x_hours, mean, color=color, linewidth=2.2, label=label)

    band(ax_combined, simple_2d, COLOR_MEAN, "Area-weighted mean", show_band=False)
    band(ax_combined, popweighted_2d, COLOR_POPWEIGHTED, "Population-weighted mean", show_band=False)
    ax_combined.set_title(title)
    ax_combined.set_ylabel(ylabel)
    ax_combined.legend(loc="best")

    band(ax_simple, simple_2d, COLOR_MEAN, "Area-weighted mean")
    ax_simple.set_title("Area-weighted: mean + ensemble range")
    ax_simple.set_ylabel(ylabel)

    band(ax_pop, popweighted_2d, COLOR_POPWEIGHTED, "Population-weighted mean")
    ax_pop.set_title("Population-weighted: mean + ensemble range")
    plt.setp(ax_pop.get_yticklabels(), visible=False)

    diff_mean = diff_2d.mean(axis=1)
    ax_diff.fill_between(x_hours, diff_2d.min(axis=1), diff_2d.max(axis=1), color=COLOR_DIFF, alpha=0.2)
    ax_diff.plot(x_hours, diff_mean, color=COLOR_DIFF, linewidth=2.2)
    ax_diff.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax_diff.set_title("Difference: population-weighted minus area-weighted (per member)")
    ax_diff.set_ylabel(f"Δ {ylabel}")

    for ax in (ax_combined, ax_simple, ax_pop, ax_diff):
        ax.set_xlabel("Lead time (hours since UTC init)")
        ax.grid(True, color="#DDDDDD", linewidth=0.6)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else zarr_path
    if not Path(path).exists():
        print(f"Error: zarr store not found at '{path}'")
        sys.exit(2)

    print(f"Opening {path} ...")
    ds = xr.open_zarr(path)
    x_hours = lead_time_hours(ds)
    member_ids = ds["ensemble"].values if "ensemble" in ds.coords else np.arange(ds.sizes["ensemble"])
    has_wind = {"u10m", "v10m"} <= set(ds.data_vars)

    output_dir.mkdir(parents=True, exist_ok=True)

    ds_gif = coarsen_spatial(ds, GIF_SPATIAL_COARSEN_FACTOR)

    print("Rendering Munich meteograms...")
    munich = nearest_point(ds, MUNICH_LAT, MUNICH_LON)

    if "t2m" in munich.data_vars:
        t2m_munich = drop_time(munich["t2m"]).transpose("lead_time", "ensemble").compute().values - 273.15
        plot_meteogram_boxplot(
            x_hours, t2m_munich, "Munich - 2m temperature ensemble meteogram", "Temperature (deg C)",
            output_dir / "meteogram_t2m_munich.png",
        )
    else:
        print("[WARN] 't2m' not present, skipping Munich temperature meteogram.")

    if has_wind:
        wind_munich = np.hypot(munich["u10m"], munich["v10m"])
        wind_munich = drop_time(wind_munich).transpose("lead_time", "ensemble").compute().values
        plot_meteogram_boxplot(
            x_hours, wind_munich, "Munich - 10m wind speed ensemble meteogram", "Wind speed (m/s)",
            output_dir / "meteogram_wind10m_munich.png",
        )
    else:
        print("[WARN] u10m/v10m not present, skipping Munich wind meteogram.")

    print("Rendering Germany temperature comparison (area-weighted vs. population-weighted)...")
    try:
        germany_area_weight, population_weight = load_germany_weights(paths.germany_population_mask_path, ds)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
    else:
        if "t2m" in ds.data_vars:
            _, spatial_dims = group_and_spatial_dims(ds["t2m"])
            simple = drop_time(global_mean(ds["t2m"], germany_area_weight, spatial_dims))
            simple = simple.transpose("lead_time", "ensemble").compute().values - 273.15
            popweighted = drop_time(global_mean(ds["t2m"], population_weight, spatial_dims))
            popweighted = popweighted.transpose("lead_time", "ensemble").compute().values - 273.15
            plot_germany_comparison(
                x_hours, simple, popweighted,
                "Germany - 2m temperature: area-weighted vs. population-weighted mean",
                "Temperature (deg C)", output_dir / "germany_t2m_simple_vs_popweighted.png",
            )
        else:
            print("[WARN] 't2m' not present, skipping Germany temperature comparison.")

    # Per-member gifs are regenerable and too large for git, so they go under
    # data/ (DVC-cached) rather than output/ (git-tracked) - see
    # ProjPaths.ensemble_gifs_path.
    gifs_dir = paths.ensemble_gifs_path
    for i, member_id in enumerate(member_ids):
        member_dir = gifs_dir / f"member_{int(member_id):02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering member {member_id} ({i + 1}/{len(member_ids)}) ...")

        if "t2m" in ds.data_vars:
            render_robinson_gif_scalar(
                ds_gif, i, member_id, "t2m", member_dir / "t2m_robinson.gif",
                cmap="RdBu_r",
                norm=mcolors.TwoSlopeNorm(vmin=T2M_VMIN, vcenter=T2M_VCENTER, vmax=T2M_VMAX),
                cbar_label="2m temperature (K)",
            )
        if has_wind:
            render_robinson_gif_wind(ds_gif, i, member_id, member_dir / "wind10m_robinson.gif")
        if "z500" in ds.data_vars:
            render_robinson_gif_scalar(
                ds_gif, i, member_id, "z500", member_dir / "z500_robinson.gif",
                cmap="viridis",
                norm=mcolors.Normalize(vmin=Z500_VMIN, vmax=Z500_VMAX),
                cbar_label="500 hPa geopotential (m^2/s^2)",
            )

    if "t2m" in ds.data_vars:
        print("Rendering ensemble-spread gif (t2m std across members)...")
        std_dir = gifs_dir / "ensemble_std"
        std_dir.mkdir(parents=True, exist_ok=True)
        render_robinson_gif_ensemble_std(
            ds_gif, "t2m", std_dir / "t2m_std_robinson.gif",
            cmap="magma", cbar_label="2m temperature ensemble std (K)",
        )

    print(f"\nDone. Meteograms written to {output_dir}/, gifs written to {gifs_dir}/")


if __name__ == "__main__":
    main()
