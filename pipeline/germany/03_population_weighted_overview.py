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
# # Population-weighted temperature: does it matter?
#
# A plain regional mean weights every grid cell by its area, so a cell over
# the Alps counts exactly as much as one over Berlin. But for things like
# energy demand, what matters is the temperature where people actually
# live. This chapter builds a population-weighting mask for Germany from
# PECD v4.2 (Pan-European Climate Database, built from NASA SEDAC 2020
# population data) and compares the resulting population-weighted mean
# against the project's standard area-weighted mean.
#
# **Input data**: the same forecast as the main "Ensemble forecast"
# chapter, not the SFNO-based downscaling experiment - **FCN3**, 8
# members, `Zero()` perturbation (`pipeline/ensemble/01_run.py`). `Zero()`
# here still produces genuine member-to-member spread because FCN3 is
# itself a stochastic model; no explicit IC/noise perturbation is added on
# top. (Contrast with the downscaling chapter's SFNO+InterpModAFNO
# forecast, which needs an explicit `Brown()` IC perturbation instead,
# since SFNO alone is deterministic - see `pipeline/downscaling/01_run.py`.)
#
# See `pipeline/germany/01_run.py` and `02_build_germany_mask.py` for how
# the mask itself is fetched (CDS `sis-energy-pecd` dataset) and regridded
# onto FCN3's native 0.25-deg grid.

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

# Germany's extent plus a small margin, so the coastline/border context
# isn't cropped tight to the data.
GERMANY_EXTENT = [5.0, 16.0, 46.5, 56.0]  # lon_min, lon_max, lat_min, lat_max

# %% [markdown]
# ## The mask
#
# Population per native (0.25 deg) grid cell, zero outside Germany's
# border by construction - see `02_build_germany_mask.py`. The
# concentration around Berlin, the Rhine-Ruhr area, and Munich is visible
# at a glance; large stretches (the former East Germany, the Alps) barely
# register.

# %%
mask_ds = xr.open_dataset(paths.germany_population_mask_path)
lat_name = next(n for n in ("lat", "latitude") if n in mask_ds.coords)
lon_name = next(n for n in ("lon", "longitude") if n in mask_ds.coords)

fig = plt.figure(figsize=(7, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(GERMANY_EXTENT, crs=ccrs.PlateCarree())
ax.coastlines(resolution="50m", linewidth=0.8, color="#444444")
ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="#444444")

# Zero-population cells (outside Germany, or uninhabited inside it) plotted
# as transparent rather than dark-purple-at-zero, so the border/coastline
# underneath does the work of showing Germany's outline.
population = mask_ds["population_weight"].where(mask_ds["population_weight"] > 0)
pcm = ax.pcolormesh(
    mask_ds[lon_name], mask_ds[lat_name], population,
    transform=ccrs.PlateCarree(), cmap="viridis", shading="auto",
)
fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.05, label="Population per grid cell")
ax.set_title("Germany population-weighting mask (PECD v4.2 / SEDAC 2020)")
fig.tight_layout()
fig.savefig(paths.germany_book_path / "population_mask.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/germany/book/population_mask.png
# :name: fig-germany-population-mask
# Population per native grid cell across Germany - the weighting applied
# below instead of a flat area-weighted mean.
# ```
#
# ## Area-weighted vs. population-weighted mean
#
# Same ensemble forecast, same 2m temperature field - only the spatial
# weighting differs. "Area-weighted" here is this project's standard
# regional-mean convention (`e2s.validation.area_weights`): it corrects for
# grid cells shrinking toward the pole, but otherwise treats every cell
# within Germany equally regardless of who lives there. The
# population-weighted mean instead uses the mask above. Computed by
# `pipeline/ensemble/04_analyse.py` via `load_germany_weights()`/
# `global_mean()` - see `e2s/validation.py` for the shared weighting logic
# used throughout this project.
#
# ```{figure} ../../output/ensemble/analysis/germany_t2m_simple_vs_popweighted.png
# :name: fig-germany-t2m-comparison
# Germany 2m temperature: area-weighted vs. population-weighted ensemble
# mean. Top: both means overlaid. Middle: each weighting's own mean plus
# its ensemble min-max range. Bottom: the per-member difference
# (population-weighted minus area-weighted) - whether that gap is stable
# across the ensemble or itself uncertain.
# ```
