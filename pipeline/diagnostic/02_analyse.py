# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python (e2s GPU)
#   language: python
#   name: e2s-gpu
# ---

# %% [markdown]
# # Diagnostic model: simulating a variable the forecast model doesn't have
#
# `01_run_solar_radiation.py` couples SFNO (prognostic) with
# `SolarRadiationAFNO6H` (diagnostic, via `earth2studio.run.diagnostic`) to
# derive `ssrd` - accumulated surface solar radiation [J/m^2] - a variable
# neither SFNO nor FCN3 forecasts on their own. The diagnostic model reads
# 24 of the prognostic model's state variables at each step (temperature,
# humidity and geopotential at several pressure levels, plus surface
# pressure and total column water vapor) and maps them to `ssrd`.
#
# This is a single deterministic run, not an ensemble - `run.diagnostic`
# has no perturbation step, so there's one trajectory, not eight.

# %%
import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import drop_time, lead_time_hours, nearest_point

paths = ProjPaths()
paths.ensure_directories()

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755
COLOR_MEAN = "#1F5C99"

ds = xr.open_zarr(paths.diagnostic_zarr_path)
x_hours = lead_time_hours(ds)

# %% [markdown]
# ## Diurnal cycle at Munich
#
# `ssrd` should track the day/night cycle - near zero overnight, positive
# during daylight hours. This is the basic sanity check for a diagnostic
# model: does its output behave like the physical quantity it claims to be,
# not just "some numbers in the right shape".

# %%
munich = nearest_point(ds, MUNICH_LAT, MUNICH_LON)
ssrd_munich = drop_time(munich["ssrd"]).compute().values

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x_hours, ssrd_munich, color=COLOR_MEAN, linewidth=2.0, marker="o", markersize=4)
ax.axhline(0, color="#AAAAAA", linewidth=0.8)
ax.set_xlabel("Lead time (hours)")
ax.set_ylabel("Accumulated solar radiation (J/m^2)")
ax.set_title("Munich - diagnosed surface solar radiation (ssrd)")
ax.grid(True, color="#DDDDDD", linewidth=0.6)
fig.tight_layout()
fig.savefig(paths.diagnostic_output_path / "ssrd_timeseries_munich.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/diagnostic/ssrd_timeseries_munich.png
# :name: fig-diagnostic-ssrd-munich
# Diagnosed solar radiation at Munich over the rollout - the diurnal cycle
# should be visible as alternating near-zero and positive stretches.
# ```
#
# ## Global snapshot
#
# One lead-time step's global `ssrd` field, to check the pattern is
# physically sensible at a glance: zero over the night side of the globe,
# positive over the day side, roughly following the solar terminator.

# %%
lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)

snapshot_step = min(2, ds.sizes["lead_time"] - 1)  # a few steps in, not the (likely all-zero) step 0
field = drop_time(ds["ssrd"]).isel(lead_time=snapshot_step).compute()
lon, lat = ds[lon_name].values, ds[lat_name].values

fig = plt.figure(figsize=(9, 4.5))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines(linewidth=0.5, color="#444444")
mesh = ax.pcolormesh(lon, lat, field.values, transform=ccrs.PlateCarree(), cmap="inferno", shading="auto")
cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Accumulated solar radiation (J/m^2)")
ax.set_title(f"Global ssrd - step {snapshot_step} ({x_hours[snapshot_step]:.0f}h)")
fig.tight_layout()
fig.savefig(paths.diagnostic_output_path / "ssrd_global_snapshot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/diagnostic/ssrd_global_snapshot.png
# :name: fig-diagnostic-ssrd-global
# Global solar radiation field at one lead-time step - the day/night
# terminator should be visible as the boundary between zero and positive
# values.
# ```
