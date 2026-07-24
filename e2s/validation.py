import re

import numpy as np
import xarray as xr

# Physically reasonable bounds per variable, matched by regex against the
# variable name. Checked in order; first match wins. Variables with no match
# are only checked for NaN/Inf, with a warning, by the caller.
#
# Geopotential (z*) is handled separately below, not in this table: a flat
# bound across all pressure levels doesn't work for it. z50's plausible
# range (~20km altitude) is roughly 4x z1000's in absolute geopotential
# terms - a bound wide enough for z50 would never catch a blown-up z1000,
# and a bound tight enough for z1000 would reject every legitimate z50
# value. See _z_bounds().
VARIABLE_BOUNDS = [
    (r"^t2m$",            (150.0, 340.0),      "K"),        # 2m temperature
    (r"^t\d{2,4}$",       (150.0, 340.0),      "K"),        # temperature @ pressure level
    (r"^(u|v)(10|100)m$", (-100.0, 100.0),     "m/s"),       # 10m/100m wind
    (r"^(u|v)\d{2,4}$",   (-150.0, 150.0),     "m/s"),       # wind @ pressure level
    (r"^q\d{2,4}$",       (0.0, 0.05),         "kg/kg"),     # specific humidity - near-surface
                                                              # saturation in warm/humid air masses
                                                              # can reach ~0.045 kg/kg
    (r"^r\d{2,4}$",       (0.0, 100.0),        "%"),         # relative humidity
    (r"^w\d{2,4}$",       (-50.0, 50.0),       "Pa/s"),      # vertical velocity
    (r"^sp$",             (40000.0, 110000.0), "Pa"),        # surface pressure
    (r"^msl$",            (85000.0, 110000.0), "Pa"),        # mean sea level pressure
    (r"^tcwv$",           (0.0, 100.0),        "kg/m^2"),    # total column water vapor
    (r"^tp\d*$",          (0.0, 2000.0),       "mm"),        # total precipitation
]

G = 9.80665  # standard gravity, m/s^2 - converts geopotential height to geopotential

# Approximate ICAO/US Standard Atmosphere geopotential height (m) by
# pressure level (hPa), used as the center of a per-level geopotential
# bounds check - see the VARIABLE_BOUNDS comment above for why this can't
# just be one more row in that table.
STD_HEIGHT_M = {
    50: 20576, 100: 16180, 150: 13608, 200: 11784, 250: 10363,
    300: 9164, 400: 7185, 500: 5574, 600: 4206, 700: 3012,
    850: 1457, 925: 762, 1000: 110,
}
# Generous tolerance around the standard-atmosphere height, in meters - for
# catching gross errors (NaN, blow-up), not enforcing climatological
# tightness. Real deviations (polar vortex, tropical warm columns, sudden
# stratospheric warming) can be a large fraction of this at the upper
# levels, and orographic extrapolation can swing z1000 by a similar amount
# at the surface.
Z_HEIGHT_MARGIN_M = 4000.0


def _z_bounds(level):
    h_std = STD_HEIGHT_M.get(level)
    if h_std is None:
        return None
    return (h_std - Z_HEIGHT_MARGIN_M) * G, (h_std + Z_HEIGHT_MARGIN_M) * G


def bounds_for(var_name):
    z_match = re.match(r"^z(\d{2,4})$", var_name)
    if z_match:
        bounds = _z_bounds(int(z_match.group(1)))
        if bounds is not None:
            return bounds, "m^2/s^2"
        return None, None  # unknown level - don't silently reuse another level's range

    for pattern, bounds, units in VARIABLE_BOUNDS:
        if re.match(pattern, var_name):
            return bounds, units
    return None, None


def group_and_spatial_dims(arr):
    group_dims = [d for d in ("time", "lead_time", "ensemble") if d in arr.dims]
    spatial_dims = [d for d in arr.dims if d not in group_dims]
    return group_dims, spatial_dims


def area_weights(ds):
    lat_name = next((n for n in ("lat", "latitude") if n in ds.coords), None)
    if lat_name is None:
        return None
    w = np.cos(np.deg2rad(ds[lat_name]))
    return w / w.mean()


def global_mean(arr, weights, spatial_dims):
    if weights is None or not spatial_dims:
        return arr.mean(dim=spatial_dims)
    return arr.weighted(weights).mean(dim=spatial_dims)


def drop_time(da):
    return da.isel(time=0, drop=True) if "time" in da.dims else da


def lead_time_hours(ds):
    """Hours elapsed since init. All init times in this project (from GFS)
    are UTC, so any "lead time" x-axis built from this is implicitly UTC-
    referenced - label it as such rather than leaving it ambiguous."""
    lt = ds["lead_time"].values
    if np.issubdtype(lt.dtype, np.timedelta64):
        return (lt / np.timedelta64(1, "h")).astype(float)
    return np.asarray(lt, dtype=float)


def nearest_point(ds, lat, lon):
    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    lon_vals = ds[lon_name].values
    target_lon = lon if lon_vals.max() <= 180 else (lon % 360)
    return ds.sel({lat_name: lat, lon_name: target_lon}, method="nearest")


def ensemble_spread_series(ds, weights, var_name):
    """Std-dev across ensemble members of the global-mean field, vs. lead_time."""
    arr = ds[var_name]
    _, spatial_dims = group_and_spatial_dims(arr)
    gmean = global_mean(arr, weights, spatial_dims)
    spread = gmean.std(dim="ensemble").compute()
    return drop_time(spread)


def load_germany_weights(mask_path, ds):
    """Load the Germany population mask built by
    pipeline/germany/02_build_germany_mask.py and derive the two weight
    fields needed to compare a plain vs. population-weighted regional mean
    with the same global_mean() helper used everywhere else: an
    area-weighted field restricted to Germany, and the population-weighted
    field itself (already zero outside Germany by construction). Both
    variants are computed by global_mean() from the exact same forecast
    array, so there's no separate model run or random seed to keep in
    sync between them - only the weighting differs.

    The mask file is regridded onto whatever grid `ds` actually uses,
    rather than assumed to match it exactly: it's built once against one
    model's native grid (pipeline/germany/02_build_germany_mask.py uses
    FCN3's), but different prognostic models can have subtly different
    grids even at the same nominal resolution - e.g. SFNO's 720-row grid
    excludes the South Pole row FCN3's 721-row grid includes - so a
    caller analyzing a different model's output (e.g.
    pipeline/downscaling/02_analyse.py, SFNO-based) would otherwise hit
    an xarray alignment error despite Germany itself being nowhere near
    the discrepancy.

    Raises FileNotFoundError if the mask hasn't been built yet (requires a
    CDS account - see pipeline/germany/01_run.py's docstring); callers
    should catch this and skip the Germany section rather than fail the
    whole stage.
    """
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Germany population mask not found at {mask_path} - run "
            "pipeline/germany/01_run.py and 02_build_germany_mask.py first "
            "(01_run.py fakes its output until a CDS account/API key is "
            "set up - see that script's docstring)."
        )
    germany = xr.open_dataset(mask_path)

    lat_name = next(n for n in ("lat", "latitude") if n in ds.coords)
    lon_name = next(n for n in ("lon", "longitude") if n in ds.coords)
    mask_lat_name = next(n for n in ("lat", "latitude") if n in germany.coords)
    mask_lon_name = next(n for n in ("lon", "longitude") if n in germany.coords)

    def snap_to_grid(da):
        da = da.rename({mask_lat_name: lat_name, mask_lon_name: lon_name}).sel(
            {lat_name: ds[lat_name].values, lon_name: ds[lon_name].values}, method="nearest",
        )
        return da.assign_coords({lat_name: ds[lat_name].values, lon_name: ds[lon_name].values})

    germany_mask = snap_to_grid(germany["germany_mask"])
    population_weight = snap_to_grid(germany["population_weight"])
    germany_area_weight = area_weights(ds).where(germany_mask, 0.0)
    return germany_area_weight, population_weight
