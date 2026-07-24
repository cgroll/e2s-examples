import re

import numpy as np

# Physically reasonable bounds per variable, matched by regex against the
# variable name. Checked in order; first match wins. Variables with no match
# are only checked for NaN/Inf, with a warning, by the caller.
VARIABLE_BOUNDS = [
    (r"^t2m$",            (150.0, 340.0),      "K"),        # 2m temperature
    (r"^t\d{2,4}$",       (150.0, 340.0),      "K"),        # temperature @ pressure level
    (r"^(u|v)(10|100)m$", (-100.0, 100.0),     "m/s"),       # 10m/100m wind
    (r"^(u|v)\d{2,4}$",   (-150.0, 150.0),     "m/s"),       # wind @ pressure level
    (r"^z\d{2,4}$",       (-2000.0, 60000.0),  "m^2/s^2"),   # geopotential
    (r"^q\d{2,4}$",       (0.0, 0.04),         "kg/kg"),     # specific humidity
    (r"^r\d{2,4}$",       (0.0, 100.0),        "%"),         # relative humidity
    (r"^w\d{2,4}$",       (-50.0, 50.0),       "Pa/s"),      # vertical velocity
    (r"^sp$",             (40000.0, 110000.0), "Pa"),        # surface pressure
    (r"^msl$",            (85000.0, 110000.0), "Pa"),        # mean sea level pressure
    (r"^tcwv$",           (0.0, 100.0),        "kg/m^2"),    # total column water vapor
    (r"^tp\d*$",          (0.0, 2000.0),       "mm"),        # total precipitation
]


def bounds_for(var_name):
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
