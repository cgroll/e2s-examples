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

from common import drop_time, lead_time_hours, nearest_point

zarr_path = "outputs/fcn3_ensemble.zarr"
output_dir = Path("analysis_output")

# Render every Nth lead_time step into the gifs (1 = every step). Bumping this
# up trades animation smoothness for a lot less render time on long rollouts.
GIF_STEP_STRIDE = 1
GIF_FRAME_DURATION = 0.15  # seconds per frame
GIF_DPI = 100

# Fixed color-scale limits so frames/members are visually comparable.
T2M_VMIN, T2M_VCENTER, T2M_VMAX = 220.0, 273.15, 320.0  # Kelvin
WIND_SPEED_MAX = 40.0  # m/s

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755

COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"


# ---------------------------------------------------------------------------
# Robinson-projection gifs
# ---------------------------------------------------------------------------
def render_robinson_gif_scalar(ds, member_index, member_id, var_name, out_path, cmap, norm, cbar_label):
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    arr = drop_time(ds[var_name].isel(ensemble=member_index))
    lon, lat = ds[lon_name].values, ds[lat_name].values

    frames = []
    for step in range(0, arr.sizes["lead_time"], GIF_STEP_STRIDE):
        data = arr.isel(lead_time=step).values
        fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5, color="#444444")
        mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
        cbar.set_label(cbar_label)
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

    frames = []
    for step in range(0, u.sizes["lead_time"], GIF_STEP_STRIDE):
        u_step = u.isel(lead_time=step).values
        v_step = v.isel(lead_time=step).values
        speed_step = np.hypot(u_step, v_step)

        fig = plt.figure(figsize=(8, 4.5), dpi=GIF_DPI)
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.5, color="#444444")
        mesh = ax.pcolormesh(
            lon, lat, speed_step, transform=ccrs.PlateCarree(),
            cmap="viridis", vmin=0, vmax=WIND_SPEED_MAX, shading="auto",
        )
        ax.quiver(
            lon[::lon_stride], lat[::lat_stride],
            u_step[::lat_stride, ::lon_stride], v_step[::lat_stride, ::lon_stride],
            transform=ccrs.PlateCarree(), color="black", scale=800, width=0.0022,
        )
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
        cbar.set_label("10m wind speed (m/s)")
        ax.set_title(f"10m wind - member {member_id} - step {step}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, duration=GIF_FRAME_DURATION, loop=0)


# ---------------------------------------------------------------------------
# Munich meteograms (ensemble boxplot per lead_time step)
# ---------------------------------------------------------------------------
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
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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

    for i, member_id in enumerate(member_ids):
        member_dir = output_dir / f"member_{int(member_id):02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        print(f"Rendering member {member_id} ({i + 1}/{len(member_ids)}) ...")

        if "t2m" in ds.data_vars:
            render_robinson_gif_scalar(
                ds, i, member_id, "t2m", member_dir / "t2m_robinson.gif",
                cmap="RdBu_r",
                norm=mcolors.TwoSlopeNorm(vmin=T2M_VMIN, vcenter=T2M_VCENTER, vmax=T2M_VMAX),
                cbar_label="2m temperature (K)",
            )
        if has_wind:
            render_robinson_gif_wind(ds, i, member_id, member_dir / "wind10m_robinson.gif")

    print(f"\nDone. Analysis charts written to {output_dir}/")


if __name__ == "__main__":
    main()
