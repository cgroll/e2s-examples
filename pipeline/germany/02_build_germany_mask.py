"""Crop the raw population mask (real PECD download, or for now the
synthetic placeholder from 01_run.py - see that script's docstring) to
Germany and regrid it onto the FCN3/SFNO native 0.25-deg grid.

Kept separate from 01_run.py so that swapping the fake mask for a real
CDS download later never touches this logic: crop + regrid is identical
either way.

Output: a small NetCDF with two fields on the native grid -
`germany_mask` (bool, True inside Germany) and `population_weight`
(population count, zero outside Germany) - consumed by
e2s.validation.load_germany_weights() in
pipeline/ensemble/04_analyse.py and pipeline/downscaling/02_analyse.py.
"""

import cartopy.io.shapereader as shpreader
import numpy as np
import shapely
import xarray as xr
from shapely.ops import unary_union

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

# Zarr store providing the canonical FCN3/SFNO native grid to regrid the
# population mask onto - any already-produced experiment output works
# equally well here, this one's just the first to exist chronologically.
TARGET_GRID_ZARR = paths.ensemble_zarr_path

raw_path = paths.germany_population_mask_raw_path
out_path = paths.germany_population_mask_path


def germany_polygon():
    """Natural Earth admin-0 country boundary for Germany - the same data
    source cartopy already uses for coastlines elsewhere in this project
    (e.g. pipeline/ensemble/04_analyse.py's Robinson gifs), so this adds
    no new dependency."""
    shp_path = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shp_path)
    matches = [r.geometry for r in reader.records() if r.attributes.get("ADM0_A3") == "DEU"]
    if not matches:
        raise RuntimeError("Germany (ADM0_A3='DEU') not found in Natural Earth admin_0_countries.")
    return unary_union(matches)


def germany_mask_on_grid(lat, lon):
    """Boolean mask, True where a grid cell center falls inside Germany's
    border. shapely.vectorized.contains does the point-in-polygon test
    over the whole grid at once, avoiding a per-cell Python loop."""
    germany = germany_polygon()
    # Natural Earth polygons use -180..180 longitude; normalize the grid's
    # longitude the same way before the point-in-polygon test, in case
    # it's stored 0..360 (common for reanalysis-derived grids).
    lon_pm180 = np.where(lon > 180, lon - 360, lon)
    lon_grid, lat_grid = np.meshgrid(lon_pm180, lat)
    return shapely.contains_xy(germany, lon_grid, lat_grid)


def main():
    print(f"Opening raw population mask {raw_path} ...")
    pop_raw = xr.open_dataset(raw_path)
    if "note" in pop_raw.attrs:
        print(f"[NOTE] {pop_raw.attrs['note']}")

    if len(pop_raw.data_vars) != 1:
        raise RuntimeError(
            f"Expected a single population variable in the raw download, found "
            f"{list(pop_raw.data_vars)} - update this script to pick the right one."
        )
    pop_var = next(iter(pop_raw.data_vars))
    pop_lat_name = next(n for n in ("lat", "latitude") if n in pop_raw.coords)
    pop_lon_name = next(n for n in ("lon", "longitude") if n in pop_raw.coords)

    print(f"Opening target grid from {TARGET_GRID_ZARR} ...")
    target = xr.open_zarr(TARGET_GRID_ZARR)
    target_lat_name = next(n for n in ("lat", "latitude") if n in target.coords)
    target_lon_name = next(n for n in ("lon", "longitude") if n in target.coords)
    target_lat = target[target_lat_name]
    target_lon = target[target_lon_name]

    # Regrid via nearest-neighbor: the source is already ~0.25 deg (same
    # nominal resolution as the target), so this is a light snap-to-grid
    # rather than a real resampling - safer than assuming exact coordinate
    # alignment between the two products.
    print("Regridding population mask onto the FCN3/SFNO native grid...")
    # .sel(method="nearest") rather than .interp(method="nearest"): xarray's
    # interp() always routes through scipy even for nearest-neighbor, which
    # isn't installed in this venv; orthogonal nearest-neighbor selection
    # via .sel() needs only pandas' Index.get_indexer, no scipy.
    population = pop_raw[pop_var].rename({pop_lat_name: "lat", pop_lon_name: "lon"}).sel(
        lat=target_lat.values, lon=target_lon.values, method="nearest",
    )
    # .sel(method="nearest") keeps the *source* grid's matched coordinate
    # values, not the query values - snap them to the target grid exactly
    # so this aligns coordinate-for-coordinate with germany_mask below.
    population = population.assign_coords(lat=target_lat.values, lon=target_lon.values)

    print("Building Germany boundary mask on the native grid...")
    germany_mask = xr.DataArray(
        germany_mask_on_grid(target_lat.values, target_lon.values),
        dims=(target_lat_name, target_lon_name),
        coords={target_lat_name: target_lat.values, target_lon_name: target_lon.values},
    )

    population_in_germany = population.rename({"lat": target_lat_name, "lon": target_lon_name})
    population_in_germany = population_in_germany.where(germany_mask, 0.0)

    n_cells = int(germany_mask.sum())
    print(f"Germany mask covers {n_cells} native grid cells; "
          f"total weighted population = {float(population_in_germany.sum()):,.0f}")
    if n_cells == 0:
        raise RuntimeError("Germany mask is empty on the native grid - check longitude convention/regridding.")

    out = xr.Dataset({
        "germany_mask": germany_mask,
        "population_weight": population_in_germany,
    })
    out.to_netcdf(out_path)
    print(f"Saved Germany population mask to {out_path}")


if __name__ == "__main__":
    main()
