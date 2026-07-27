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
zarr_path = paths.downscaling_zarr_path
output_dir = paths.downscaling_analysis_path

GIF_DPI = 100

# Block-average the lat/lon grid before rendering gifs - see
# pipeline/ensemble/04_analyse.py's GIF_SPATIAL_COARSEN_FACTOR docstring for
# why this is visually lossless at gif resolution.
GIF_SPATIAL_COARSEN_FACTOR = 4

# Matches InterpModAFNO's num_interp_steps default in 01_run.py - the
# spacing, in hours, between two native SFNO outputs. Every lead_time step
# at this spacing is the base model's own 6h prediction, passed through
# unmodified; everything in between is AFNO-interpolated. See
# earth2studio's interpmodafno.py _default_generator: `yield (x1, coords)`
# for the native step vs. `_interpolate(...)` for the steps in between.
NATIVE_STEP_HOURS = 6

# Frame duration for the dense (1h-cadence) gif - matches the pacing used
# for the main ensemble's 6h-native gifs in pipeline/ensemble/04_analyse.py
# (GIF_FRAME_DURATION = 0.15s), so it stays comfortably watchable. The
# native-only gif's frames each represent NATIVE_STEP_HOURS times as much
# simulated time, so its frame duration is scaled up by the same factor -
# both gifs then reach the same simulated hour at the same wall-clock time,
# not just finish at the same total length.
DENSE_FRAME_DURATION = 0.15  # seconds per simulated hour, dense gif
NATIVE_FRAME_DURATION = DENSE_FRAME_DURATION * NATIVE_STEP_HOURS

# Fixed color-scale limits so frames/members are visually comparable -
# matches pipeline/ensemble/04_analyse.py's T2M scale.
T2M_VMIN, T2M_VCENTER, T2M_VMAX = 220.0, 273.15, 320.0  # Kelvin

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755

COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"
COLOR_NATIVE = "#C9622A"  # highlights native-SFNO-step boxes/frames
COLOR_POPWEIGHTED = "#2E8B57"  # distinct from COLOR_NATIVE - different meaning, same chapter
COLOR_DIFF = "#6A4C93"


# ---------------------------------------------------------------------------
# Robinson-projection gif: t2m only, over a given subset of lead_time steps
# ---------------------------------------------------------------------------
def render_robinson_gif_t2m(ds, member_index, member_id, lead_time_indices, out_path, frame_duration, hours, tag_frames):
    """tag_frames=True labels each frame as native-SFNO-step or
    AFNO-interpolated in its title (used for the dense gif, which mixes
    both); False assumes every rendered step is native (used for the
    native-only gif, where that's true by construction)."""
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    arr = drop_time(ds["t2m"].isel(ensemble=member_index))
    lon, lat = ds[lon_name].values, ds[lat_name].values

    # Figure/axes/mesh created once and reused across frames - see
    # pipeline/ensemble/04_analyse.py's render_robinson_gif_scalar for why.
    fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color="#444444")

    mesh = None
    frames = []
    for step in lead_time_indices:
        data = arr.isel(lead_time=step).values
        if mesh is None:
            mesh = ax.pcolormesh(
                lon, lat, data, transform=ccrs.PlateCarree(), cmap="RdBu_r",
                norm=mcolors.TwoSlopeNorm(vmin=T2M_VMIN, vcenter=T2M_VCENTER, vmax=T2M_VMAX),
                shading="auto",
            )
            cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
            cbar.set_label("2m temperature (K)")
        else:
            mesh.set_array(data.ravel())

        hour = hours[step]
        if tag_frames:
            is_native = round(hour) % NATIVE_STEP_HOURS == 0
            tag = "native SFNO step" if is_native else "AFNO-interpolated step"
        else:
            tag = "native SFNO step"
        ax.set_title(f"t2m - member {member_id} - hour {hour:.0f} ({tag})")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

    plt.close(fig)
    imageio.mimsave(out_path, frames, duration=frame_duration, loop=0)


# ---------------------------------------------------------------------------
# Munich meteogram (ensemble boxplot per lead_time step)
# ---------------------------------------------------------------------------
def coarsen_spatial(ds, factor):
    if factor <= 1:
        return ds
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    return ds.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean()


def plot_meteogram_boxplot(x_hours, data_2d, title, ylabel, out_path, highlight_mask=None):
    """data_2d: shape (n_lead_time, n_ensemble). If highlight_mask is given
    (one bool per lead_time step), boxes at True positions are colored to
    mark native-SFNO steps among the AFNO-interpolated ones."""
    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(
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
    if highlight_mask is not None:
        for box, is_native in zip(bp["boxes"], highlight_mask):
            if is_native:
                box.set_facecolor(COLOR_NATIVE)
                box.set_alpha(0.8)

    step = max(1, data_2d.shape[0] // 20)
    ax.set_xticks(np.arange(0, data_2d.shape[0], step))
    ax.set_xticklabels([f"{h:.0f}" for h in x_hours[::step]], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Lead time (hours since UTC init)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)

    if highlight_mask is not None and any(highlight_mask):
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MEMBER, alpha=0.5, edgecolor="#444444", label="AFNO-interpolated step"),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_NATIVE, alpha=0.8, edgecolor="#444444", label="Native SFNO step"),
        ]
        ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_germany_comparison(x_hours, simple_2d, popweighted_2d, title, ylabel, out_path):
    """simple_2d/popweighted_2d: shape (n_lead_time, n_ensemble) - both
    computed from the same forecast array via global_mean() with
    different weight fields, member-for-member aligned (ensemble index i
    is the same underlying forecast draw in both), so their difference
    below is a genuine per-member difference, not a comparison across
    unrelated samples; see e2s.validation.load_germany_weights.

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
    hours = lead_time_hours(ds)
    native_mask = np.round(hours) % NATIVE_STEP_HOURS == 0
    native_indices = np.where(native_mask)[0]
    dense_indices = np.arange(len(hours))
    member_ids = ds["ensemble"].values if "ensemble" in ds.coords else np.arange(ds.sizes["ensemble"])

    output_dir.mkdir(parents=True, exist_ok=True)
    ds_gif = coarsen_spatial(ds, GIF_SPATIAL_COARSEN_FACTOR)

    print("Rendering Munich meteograms (dense vs. native-only)...")
    munich = nearest_point(ds, MUNICH_LAT, MUNICH_LON)

    if "t2m" in munich.data_vars:
        t2m_munich = drop_time(munich["t2m"]).transpose("lead_time", "ensemble").compute().values - 273.15
        plot_meteogram_boxplot(
            hours, t2m_munich,
            "Munich - 2m temperature, hourly (native SFNO steps highlighted)",
            "Temperature (deg C)", output_dir / "meteogram_t2m_munich_dense.png",
            highlight_mask=native_mask,
        )
        plot_meteogram_boxplot(
            hours[native_indices], t2m_munich[native_indices],
            "Munich - 2m temperature, native 6h SFNO steps only",
            "Temperature (deg C)", output_dir / "meteogram_t2m_munich_native.png",
        )
    else:
        print("[WARN] 't2m' not present, skipping Munich meteograms.")

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
                hours, simple, popweighted,
                "Germany - 2m temperature: area-weighted vs. population-weighted mean (hourly)",
                "Temperature (deg C)", output_dir / "germany_t2m_simple_vs_popweighted.png",
            )
        else:
            print("[WARN] 't2m' not present, skipping Germany temperature comparison.")

    # Per-member gifs are regenerable and too large for git, so they go under
    # data/ (DVC-cached) rather than output/ (git-tracked) - see
    # ProjPaths.downscaling_gifs_path.
    gifs_dir = paths.downscaling_gifs_path
    for i, member_id in enumerate(member_ids):
        member_dir = gifs_dir / f"member_{int(member_id):02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering member {member_id} ({i + 1}/{len(member_ids)}) ...")

        if "t2m" not in ds.data_vars:
            continue
        render_robinson_gif_t2m(
            ds_gif, i, member_id, dense_indices, member_dir / "t2m_robinson_dense.gif",
            frame_duration=DENSE_FRAME_DURATION, hours=hours, tag_frames=True,
        )
        render_robinson_gif_t2m(
            ds_gif, i, member_id, native_indices, member_dir / "t2m_robinson_native.gif",
            frame_duration=NATIVE_FRAME_DURATION, hours=hours, tag_frames=False,
        )

    print(f"\nDone. Meteograms written to {output_dir}/, gifs written to {gifs_dir}/")


if __name__ == "__main__":
    main()
