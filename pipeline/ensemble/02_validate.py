import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import (
    area_weights,
    bounds_for,
    drop_time,
    ensemble_spread_series,
    global_mean,
    group_and_spatial_dims,
    lead_time_hours,
)

paths = ProjPaths()
paths.ensure_directories()
zarr_path = paths.ensemble_zarr_path
output_dir = paths.ensemble_validation_path
tables_dir = paths.ensemble_validation_tables_path

# Pressure-level variables that must be monotonic across levels: geopotential
# height increases with altitude, and altitude increases as pressure decreases,
# so e.g. z500 > z700 > z850 must hold almost everywhere.
CONSISTENCY_PREFIXES = ["z"]
CONSISTENCY_TOLERANCE = 0.01  # fraction of grid points allowed to violate ordering, per snapshot

# Max allowed change in the (area-weighted, global-mean) value between
# consecutive lead_time steps, per variable, in native units.
STEP_JUMP_LIMITS = {
    "t2m": 3.0,
    "msl": 300.0,
    "tcwv": 2.0,
    "z500": 300.0,
}

# Baseline (step-0) referenced drift/blow-up tolerances.
MASS_DRIFT_TOLERANCE = 0.01   # max fractional drift in global-mean msl/sp vs. step 0
KE_GROWTH_FACTOR = 5.0        # global-mean 10m kinetic energy must not exceed this multiple of step 0

# Cross-ensemble spread sanity bounds (ratio of spread at step i to spread at step 0).
SPREAD_COLLAPSE_RATIO = 0.1
SPREAD_EXPLOSION_RATIO = 20.0
SPREAD_VARIABLE = "t2m"


def _lead_time_frame(hours):
    return pd.DataFrame({"lead_time": np.arange(len(hours)), "lead_time_hours": hours})


def pressure_level_vars(ds, prefix):
    import re
    pattern = re.compile(rf"^{prefix}(\d{{2,4}})$")
    levels = [(int(m.group(1)), name) for name in ds.data_vars if (m := pattern.match(name))]
    return sorted(levels)


# ---------------------------------------------------------------------------
# Table 1: standalone variable bounds - one row per (variable, ensemble, lead_time)
# ---------------------------------------------------------------------------
def build_standalone_variable_table(ds, hours):
    lt = _lead_time_frame(hours)
    rows = []

    for var_name in ds.data_vars:
        arr = ds[var_name]
        if "lead_time" not in arr.dims or "ensemble" not in arr.dims:
            continue

        bounds, units = bounds_for(var_name)
        if bounds is None:
            print(f"[WARN] No bounds defined for '{var_name}' - only checking for NaN/Inf.")
            lo, hi = -np.inf, np.inf
        else:
            lo, hi = bounds

        _, spatial_dims = group_and_spatial_dims(arr)
        n_total = int(np.prod([arr.sizes[d] for d in spatial_dims])) if spatial_dims else 1

        bad_mask = arr.isnull() | (arr < lo) | (arr > hi)
        n_bad = bad_mask.sum(dim=spatial_dims) if spatial_dims else bad_mask.astype(int)
        vmin = arr.min(dim=spatial_dims) if spatial_dims else arr
        vmax = arr.max(dim=spatial_dims) if spatial_dims else arr

        snap = drop_time(xr.Dataset({"n_bad_points": n_bad, "min_value": vmin, "max_value": vmax}).compute())
        df = snap.to_dataframe().reset_index()
        df["variable"] = var_name
        df["units"] = units or "?"
        df["n_total_points"] = n_total
        df["violating_fraction"] = df["n_bad_points"] / n_total
        df["valid"] = df["n_bad_points"] == 0
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=[
            "variable", "ensemble", "lead_time", "lead_time_hours", "units",
            "min_value", "max_value", "n_bad_points", "n_total_points", "violating_fraction", "valid",
        ])

    table = pd.concat(rows, ignore_index=True).merge(lt, on="lead_time", how="left")
    cols = ["variable", "ensemble", "lead_time", "lead_time_hours", "units",
            "min_value", "max_value", "n_bad_points", "n_total_points", "violating_fraction", "valid"]
    return table[cols].sort_values(["variable", "ensemble", "lead_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 2: cross-variable consistency - one row per (check, ensemble, lead_time)
# ---------------------------------------------------------------------------
def build_cross_variable_consistency_table(ds, hours):
    lt = _lead_time_frame(hours)
    rows = []

    for prefix in CONSISTENCY_PREFIXES:
        levels = pressure_level_vars(ds, prefix)
        for (p_low, var_low), (p_high, var_high) in zip(levels, levels[1:]):
            # p_low < p_high => var_low is the higher-altitude level and must
            # have the larger geopotential almost everywhere, per snapshot.
            lower, higher = ds[var_low], ds[var_high]
            _, spatial_dims = group_and_spatial_dims(lower)
            n_total = int(np.prod([lower.sizes[d] for d in spatial_dims])) if spatial_dims else 1

            n_violating = (lower <= higher).sum(dim=spatial_dims) if spatial_dims else (lower <= higher).astype(int)
            n_violating = drop_time(n_violating.compute())

            df = n_violating.to_dataframe(name="n_violating_points").reset_index()
            df["check"] = f"{var_low}_gt_{var_high}"
            df["var_low"] = var_low
            df["var_high"] = var_high
            df["n_total_points"] = n_total
            df["violating_fraction"] = df["n_violating_points"] / n_total
            df["tolerance"] = CONSISTENCY_TOLERANCE
            df["valid"] = df["violating_fraction"] <= CONSISTENCY_TOLERANCE
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=[
            "check", "var_low", "var_high", "ensemble", "lead_time", "lead_time_hours",
            "n_violating_points", "n_total_points", "violating_fraction", "tolerance", "valid",
        ])

    table = pd.concat(rows, ignore_index=True).merge(lt, on="lead_time", how="left")
    cols = ["check", "var_low", "var_high", "ensemble", "lead_time", "lead_time_hours",
            "n_violating_points", "n_total_points", "violating_fraction", "tolerance", "valid"]
    return table[cols].sort_values(["check", "ensemble", "lead_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 3: cross-time consistency - one row per (check, ensemble, transition)
# ---------------------------------------------------------------------------
def _series_from_gmean(gmean):
    return drop_time(gmean).to_dataframe(name="value").reset_index()


def _transition_rows(check, metric, values_df, from_idx, to_idx, threshold, metric_fn, valid_fn):
    v_from = values_df[values_df["lead_time"] == from_idx][["ensemble", "value"]].rename(columns={"value": "value_from"})
    v_to = values_df[values_df["lead_time"] == to_idx][["ensemble", "value"]].rename(columns={"value": "value_to"})
    merged = v_from.merge(v_to, on="ensemble")
    merged["check"] = check
    merged["metric"] = metric
    merged["lead_time_from"] = from_idx
    merged["lead_time_to"] = to_idx
    merged["metric_value"] = metric_fn(merged["value_from"], merged["value_to"])
    merged["threshold"] = threshold
    merged["valid"] = valid_fn(merged["metric_value"], threshold)
    return merged


def build_cross_time_consistency_table(ds, weights, hours):
    lt = _lead_time_frame(hours)
    lt_hours = dict(zip(lt["lead_time"], lt["lead_time_hours"]))
    n_lead = len(lt)
    rows = []

    delta_fn = lambda f, t: t - f
    delta_valid = lambda m, thr: m.abs() <= thr
    ratio_fn = lambda f, t: t / f
    ratio_valid = lambda m, thr: m <= thr
    frac_drift_fn = lambda f, t: (t - f) / f
    frac_drift_valid = lambda m, thr: m.abs() <= thr

    for var_name, max_delta in STEP_JUMP_LIMITS.items():
        if var_name not in ds.data_vars or "lead_time" not in ds[var_name].dims:
            continue
        _, spatial_dims = group_and_spatial_dims(ds[var_name])
        gmean = global_mean(ds[var_name], weights, spatial_dims).compute()
        values_df = _series_from_gmean(gmean)
        for i in range(1, n_lead):
            rows.append(_transition_rows(
                f"step_jump:{var_name}", var_name, values_df, i - 1, i, max_delta, delta_fn, delta_valid
            ))

    pressure_var = next((v for v in ("msl", "sp") if v in ds.data_vars), None)
    if pressure_var is not None:
        _, spatial_dims = group_and_spatial_dims(ds[pressure_var])
        gmean = global_mean(ds[pressure_var], weights, spatial_dims).compute()
        values_df = _series_from_gmean(gmean)
        for i in range(1, n_lead):
            rows.append(_transition_rows(
                f"mass_conservation:{pressure_var}", pressure_var, values_df, 0, i,
                MASS_DRIFT_TOLERANCE, frac_drift_fn, frac_drift_valid,
            ))
    else:
        print("[WARN] No 'msl' or 'sp' variable found, skipping mass conservation check.")

    if {"u10m", "v10m"} <= set(ds.data_vars):
        ke = 0.5 * (ds["u10m"] ** 2 + ds["v10m"] ** 2)
        _, spatial_dims = group_and_spatial_dims(ke)
        gmean = global_mean(ke, weights, spatial_dims).compute()
        values_df = _series_from_gmean(gmean)
        for i in range(1, n_lead):
            rows.append(_transition_rows(
                "energy_blowup:kinetic_energy", "kinetic_energy", values_df, 0, i,
                KE_GROWTH_FACTOR, ratio_fn, ratio_valid,
            ))
    else:
        print("[WARN] No 'u10m'/'v10m' variables found, skipping kinetic-energy blow-up check.")

    if not rows:
        return pd.DataFrame(columns=[
            "check", "metric", "ensemble", "lead_time_from", "lead_time_to",
            "lead_time_from_hours", "lead_time_to_hours",
            "value_from", "value_to", "metric_value", "threshold", "valid",
        ])

    table = pd.concat(rows, ignore_index=True)
    table["lead_time_from_hours"] = table["lead_time_from"].map(lt_hours)
    table["lead_time_to_hours"] = table["lead_time_to"].map(lt_hours)
    cols = ["check", "metric", "ensemble", "lead_time_from", "lead_time_to",
            "lead_time_from_hours", "lead_time_to_hours",
            "value_from", "value_to", "metric_value", "threshold", "valid"]
    return table[cols].sort_values(["check", "ensemble", "lead_time_to"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 4: cross-ensemble consistency - one row per (check, lead_time)
#
# Only rule implemented so far: ensemble spread (std-dev across members of the
# global-mean field) shouldn't collapse toward zero or blow up over the
# rollout. Grain is per lead_time, not per member, since the check is a
# property of the ensemble as a whole; an `ensemble` column is included for
# schema symmetry with the other tables and for future per-member outlier
# checks (e.g. "member i is >3 std from the ensemble mean").
#
# The ratio is measured against the spread at the first lead_time where it
# becomes nonzero, not lead_time 0: with an unperturbed IC (e.g. Zero()
# perturbation), spread at lead_time 0 is exactly 0 by construction, which
# isn't a collapse - it's the shared analysis state before any model-internal
# stochasticity has had a chance to act. Lead times before that reference
# are the expected zero-spread onset period and are treated as trivially
# valid rather than compared against a degenerate zero baseline.
# ---------------------------------------------------------------------------
def build_cross_ensemble_consistency_table(ds, weights, hours, var_name=SPREAD_VARIABLE):
    lt = _lead_time_frame(hours)

    if var_name not in ds.data_vars or "ensemble" not in ds[var_name].dims:
        print(f"[WARN] Cannot compute cross-ensemble consistency for '{var_name}'.")
        return pd.DataFrame(columns=[
            "check", "metric", "ensemble", "lead_time", "lead_time_hours",
            "spread", "ref_spread", "ratio", "valid",
        ])

    spread = ensemble_spread_series(ds, weights, var_name)
    spread_df = spread.to_dataframe(name="spread").reset_index().sort_values("lead_time").reset_index(drop=True)

    nonzero = spread_df[spread_df["spread"] > 0]
    ref_lead_time = int(nonzero["lead_time"].iloc[0]) if not nonzero.empty else None
    ref = float(nonzero["spread"].iloc[0]) if not nonzero.empty else 0.0

    spread_df["check"] = f"ensemble_spread:{var_name}"
    spread_df["metric"] = f"std_global_mean_{var_name}"
    spread_df["ensemble"] = "aggregate"
    spread_df["ref_spread"] = ref

    if ref_lead_time is None:
        # Spread never leaves zero across the whole rollout - genuine collapse.
        spread_df["ratio"] = np.inf
        spread_df["valid"] = False
    else:
        spread_df["ratio"] = spread_df["spread"] / ref
        spread_df["valid"] = spread_df["ratio"].between(SPREAD_COLLAPSE_RATIO, SPREAD_EXPLOSION_RATIO)
        spread_df.loc[spread_df["lead_time"] < ref_lead_time, "valid"] = True

    table = spread_df.merge(lt, on="lead_time", how="left")
    cols = ["check", "metric", "ensemble", "lead_time", "lead_time_hours", "spread", "ref_spread", "ratio", "valid"]
    return table[cols].sort_values("lead_time").reset_index(drop=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else zarr_path
    if not Path(path).exists():
        print(f"Error: zarr store not found at '{path}'")
        sys.exit(2)

    print(f"Opening {path} ...")
    ds = xr.open_zarr(path)
    hours = lead_time_hours(ds)
    # Normalize lead_time to a plain 0..n-1 index so it merges cleanly across
    # tables (the native coordinate is often a timedelta64, which doesn't
    # compare/merge against the plain integer index the tables key on).
    ds = ds.assign_coords(lead_time=np.arange(len(hours)))
    weights = area_weights(ds)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Building standalone variable bounds table...")
    standalone = build_standalone_variable_table(ds, hours)
    standalone.to_csv(tables_dir / "standalone_variable.csv", index=False)

    print("Building cross-variable consistency table...")
    cross_variable = build_cross_variable_consistency_table(ds, hours)
    cross_variable.to_csv(tables_dir / "cross_variable_consistency.csv", index=False)

    print("Building cross-time consistency table...")
    cross_time = build_cross_time_consistency_table(ds, weights, hours)
    cross_time.to_csv(tables_dir / "cross_time_consistency.csv", index=False)

    print("Building cross-ensemble consistency table...")
    cross_ensemble = build_cross_ensemble_consistency_table(ds, weights, hours)
    cross_ensemble.to_csv(tables_dir / "cross_ensemble_consistency.csv", index=False)

    print(f"\nTables written to {tables_dir}/")

    tables = {
        "standalone_variable": standalone,
        "cross_variable_consistency": cross_variable,
        "cross_time_consistency": cross_time,
        "cross_ensemble_consistency": cross_ensemble,
    }

    print("\n========== Summary ==========")
    all_passed = True
    for name, table in tables.items():
        if table.empty:
            print(f"{name:28s} SKIPPED (no data)")
            continue
        n_invalid = int((~table["valid"]).sum())
        passed = n_invalid == 0
        all_passed &= passed
        print(f"{name:28s} {'PASS' if passed else 'FAIL'}  ({n_invalid}/{len(table)} rows invalid)")

    if all_passed:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nOne or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
