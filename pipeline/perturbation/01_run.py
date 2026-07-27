"""Compare IC-perturbation strategies for SFNO, with per-step physical-
bounds validation (blow-up detection) and output cropped to the North-
Atlantic-European (NAE) region (Grams et al.: 80W-40E, 30-90N) to keep
storage lean.

Why this exists: pipeline/downscaling/01_run.py wraps SFNO in
InterpModAFNO and perturbs the IC with Brown(noise_amplitude=0.05) - and
that combination blows up from the very first native SFNO step (t2m/z500/
u10m/v10m all diverge to unphysical magnitudes; see the "Population-
weighted temperature"/downscaling-chapter investigation this script
followed from). This experiment isolates the question "does SFNO itself
tolerate IC perturbation, and at what amplitude" from the added
complexity of the interpolation wrapper, by running bare SFNO (no
InterpModAFNO) directly.

Two questions this sweep answers, in this order:

1. Is SFNO's own deterministic (unperturbed) rollout even reliably valid
   across different times of year, before any perturbation is added at
   all? Answered by the `ic_robustness_*` configs: four single (not
   ensemble - every member would be bit-identical anyway, since SFNO has
   no internal stochasticity and Zero() adds nothing) Zero()-perturbation
   forecasts, one per season, from four different initial-condition
   dates. This runs first (see the book chapter ordering) because
   everything downstream implicitly assumes the answer is "yes, modulo a
   known tcwv idiosyncrasy" - worth establishing directly rather than
   inferring from a single date.
2. Given that baseline, which perturbation strategies stay within
   physical bounds well enough to be usable for ensemble generation? Two
   families are tested, both now calibrated per-variable rather than with
   one flat amplitude (see e2s.perturbation.ScaledBrownPerturbation for
   why the original flat-amplitude version was abandoned - the same
   module pipeline/downscaling/01_run.py now uses too): `brown_*` (all
   73 variables, three intensities) and `z500_brown_*` (z500 only, same
   three intensities, via e2s.perturbation.SingleVariablePerturbation) -
   plus `bred_vector` (grows the perturbation via the model itself, not
   a static offset).
   Gaussian is dropped from this sweep: it failed for exactly the same
   flat-amplitude reason Brown did in the earlier version of this
   experiment, and the scaled-amplitude fix that matters here is
   Brown-specific (Gaussian's IID-per-gridpoint noise doesn't have a
   spatial reddening parameter to calibrate the same way).

Per-step validation: earth2studio's run.ensemble() applies its
perturbation exactly once and owns the rollout loop opaquely (see
earth2studio/run.py: perturbation is called once before
prognostic.create_iterator(...) starts) - it can't be paused to inspect
state mid-rollout. So every config here goes through a hand-rolled
autoregressive loop (calling model(x, coords) directly, one step at a
time) instead of run.ensemble(), the same pattern
pipeline/perturbation's earlier FCN3 version used only for its
per-step-injection config. After every step, the model's FULL global
state (all 73 variables) is scored against e2s.validation's physical-
plausibility bounds - cheap (just min/max/violating-fraction per
variable, done on-GPU) - so a violation is caught wherever it first
appears, even though only t2m/u10m/v10m/z500 over the NAE region are
actually persisted as field data.

Deliberately NOT a hard stop-on-violation gate: an earlier version of
this script froze a member's rollout the moment any variable left its
bounds, but that turned out to fire on step 1 for every config -
including the unperturbed Zero() control - because SFNO's raw tcwv
output is mildly negative (down to ~-5.5 kg/m2) across ~7% of the global
grid even with no perturbation at all, unlike FCN3 (which passes this
exact bounds table with zero violations - see
pipeline/ensemble/02_validate.py). A single-bit gate can't distinguish
that baseline idiosyncrasy from genuine catastrophic divergence. So
instead every member runs its full rollout regardless, and every step's
metrics are written to CSVs per config: per-variable bounds
(violating_fraction, min, max - same schema as 02_validate.py's
build_standalone_variable_table), cross-variable z-level ordering
consistency (same schema as build_cross_variable_consistency_table - a
field can stay within its own bounds while its ordering relative to
neighboring levels still inverts, which the bounds check alone can't
see), and global-mean timeseries (t2m/msl/sp/tcwv/z500/kinetic energy,
enough to reconstruct 02_validate.py's cross-*time* checks - step-jump
limits, mass drift, KE growth - post-hoc). Which members/steps to treat
as "blown up" is a post-hoc analysis question (02_analyse.py), not a
during-inference one.

Member 0 is always run with Zero() perturbation regardless of config - a
config-agnostic control trajectory (SFNO is deterministic without
perturbation, so this is the same physical forecast every time) to
anchor comparisons (meteograms, gifs) against, even for configs where
every other member is perturbed. The `ic_robustness_*` and `zero`
configs use Zero() for every member anyway, so this rule is a no-op for
them - they're already 100% control.

Per-step injection (like the FCN3 script's per_step_brown) is
deliberately left out of this first pass - all configs below are IC-only.

bred_vector was added after the fact, once zero/brown/gaussian all
turned out to diverge regardless of (flat) amplitude: unlike those, it
doesn't add a static offset in raw physical units - it grows the
perturbation by running the model itself for `integration_steps`
internal steps on a lightly-seeded start, then rescales the result to
`noise_amplitude` relative to the state's own norm (see
earth2studio.perturbation.BredVector.__call__: `gamma = norm(x) /
norm(x + dx)`). That self-relative scaling is the whole point - it
doesn't force one flat amplitude onto all 73 variables regardless of
their physical scale, the same problem ScaledBrownPerturbation fixes for
Brown a different way (explicit per-variable calibration instead of
implicit self-normalization). Seeded with Brown(noise_amplitude=0.002) -
the smallest, least-catastrophic amplitude from the original flat-
amplitude sweep - since the seed still has to survive 20 internal model
steps before it's rescaled into the final perturbation; a seed that
immediately diverges would just bred a diverged direction.
HemisphericCentredBredVector (the HENS-style variant used in NVIDIA's
"Huge Ensembles" paper) was considered too, but it exposes a generator-
based API (`create_generator`), not the simple `(x, coords) -> (x,
coords)` interface every other perturbation method here uses - adapting
the manual rollout loop for that is a bigger change, left for later if
bred_vector itself doesn't pan out.
"""

import re

import numpy as np
import pandas as pd
import torch

from earth2studio.data import GFS, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO
from earth2studio.perturbation import Brown, BredVector, Zero
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array

from e2s.paths import ProjPaths
from e2s.perturbation import ScaledBrownPerturbation, SingleVariablePerturbation, compute_variable_scales
from e2s.validation import bounds_for

paths = ProjPaths()
paths.ensure_directories()

# --- Compat shim: see pipeline/downscaling/01_run.py's identical block for
# why makani (pinned commit d473b054) needs this patched onto SFNO.load_model.
import makani.models.model_registry as _makani_model_registry

_original_get_model = _makani_model_registry.get_model


def _get_model_compat(params, *args, **kwargs):
    if not hasattr(params, "img_shape_x_resampled"):
        params.img_shape_x_resampled = params.img_shape_x
    if not hasattr(params, "img_shape_y_resampled"):
        params.img_shape_y_resampled = params.img_shape_y
    return _original_get_model(params, *args, **kwargs)


_makani_model_registry.get_model = _get_model_compat

N_ENSEMBLE = 4
N_STEPS = 20  # native 6h SFNO steps -> 20 = 5 days
START_DATE = "2026-07-23T00:00:00"  # UTC - GFS timestamps are always UTC, matches the other experiments' START_DATE

# North-Atlantic-European region (Grams et al.): 80W-40E, 30-90N. Only
# used to crop what gets *persisted* below - the bounds check runs on the
# full global state regardless, so a blow-up outside this box is still
# caught (see check_bounds).
NAE_LON_MIN, NAE_LON_MAX = -80.0, 40.0  # deg, -180..180 convention
NAE_LAT_MIN, NAE_LAT_MAX = 30.0, 90.0

STORED_VARIABLES = ["t2m", "u10m", "v10m", "z500"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SFNO model weights...")
package = SFNO.load_default_package()
model = SFNO.load_model(package).to(device)

data = GFS()

# Precomputed once (static across the whole sweep - same model, same
# variable list every step) rather than re-resolved via bounds_for()'s
# regex matching on every single step.
VAR_BOUNDS = {name: bounds_for(name)[0] for name in model.input_coords()["variable"]}

# Geopotential pressure levels, ascending by pressure (= descending by
# altitude) - z50 is the highest-altitude level, z1000 the lowest. Used
# for the cross-variable consistency check below: z at a given level must
# be less than z at the next-lower pressure (next-higher altitude) level
# almost everywhere, same convention as pipeline/ensemble/02_validate.py's
# CONSISTENCY_PREFIXES=["z"] check (there computed post-hoc from a saved
# zarr; here computed live per step, since the full per-level fields
# aren't otherwise persisted - only z500 is in STORED_VARIABLES).
Z_LEVELS = sorted(
    int(m.group(1)) for name in model.input_coords()["variable"]
    if (m := re.match(r"^z(\d+)$", name))
)


def make_scaled_brown(intensity):
    """Config builder: computes variable_scales from that config's own
    fetched IC (x0, coords0), not a module-level constant - see
    run_config, which calls builders after fetch_data returns."""
    def builder(x0, coords0):
        return ScaledBrownPerturbation(compute_variable_scales(x0, coords0), intensity)
    return builder


def make_z500_scaled_brown(intensity):
    def builder(x0, coords0):
        scales = compute_variable_scales(x0, coords0)
        return SingleVariablePerturbation("z500", ScaledBrownPerturbation(scales, intensity))
    return builder


def make_bred_vector():
    def builder(x0, coords0):
        return BredVector(
            model=model, noise_amplitude=0.05,
            seeding_perturbation_method=Brown(noise_amplitude=0.002),
        )
    return builder


# Four single (N_ENSEMBLE=1), unperturbed initial-condition dates, one per
# season - see module docstring's point (1). All in the past relative to
# this project's "today" so GFS has analysis data for them; 2026-07-23
# doubles as this sweep's main date elsewhere (`zero`, `brown_*`,
# `z500_brown_*`, `bred_vector` all use it too).
IC_ROBUSTNESS_DATES = {
    "ic_robustness_winter": "2026-01-15T00:00:00",
    "ic_robustness_spring": "2026-04-15T00:00:00",
    "ic_robustness_summer": "2026-07-23T00:00:00",
    "ic_robustness_autumn": "2025-10-15T00:00:00",
}

# Every entry is a builder: (x0, coords0) -> Perturbation. Even the
# configs that don't need the IC (Zero, bred_vector) take the same
# signature so run_config can treat every config uniformly - see
# run_config, which fetches data once per config, then calls the builder.
CONFIGS = {
    **{name: (lambda x0, coords0: Zero()) for name in IC_ROBUSTNESS_DATES},
    "zero": lambda x0, coords0: Zero(),
    "brown_0.05": make_scaled_brown(0.05),
    "brown_0.01": make_scaled_brown(0.01),
    "brown_0.002": make_scaled_brown(0.002),
    "z500_brown_0.05": make_z500_scaled_brown(0.05),
    "z500_brown_0.01": make_z500_scaled_brown(0.01),
    "z500_brown_0.002": make_z500_scaled_brown(0.002),
    "bred_vector": make_bred_vector(),
}

START_DATE_OVERRIDES = dict(IC_ROBUSTNESS_DATES)
N_ENSEMBLE_OVERRIDES = {name: 1 for name in IC_ROBUSTNESS_DATES}


def nae_crop_masks(lat, lon):
    """Boolean masks selecting the NAE box on SFNO's native 0..360 grid.
    Lon is 0..360 here, and -80..40 straddles the 0/360 seam (maps to the
    union of native 280..360 and 0..40) - normalize to -180..180 before
    comparing, or a plain min/max slice would silently select the wrong
    (complementary) region.

    Deliberately kept in natural ascending storage order (280..360 comes
    *after* 0..40 in the stored coordinate, not before) rather than
    reordered to be geographically contiguous: xarray requires a
    monotonic coordinate for .sel(method="nearest") (used by the Munich
    meteograms below), and 0 < 40 < 280 < 360 satisfies that, even though
    it isn't geographically contiguous. An earlier version of this
    function reordered the stored coordinate to fix that non-contiguity -
    which broke .sel() with a "must be monotonic" error, and also only
    half-fixed the actual problem (see 02_analyse.py's
    _geographic_lon_order): pcolormesh renders consecutive array entries
    as adjacent, so plotting this array in its stored (ascending but
    seam-split) order draws one giant quad across the 240 degree gap - a
    plot-time reordering, not a storage-time one, is the correct fix."""
    lat_mask = (lat >= NAE_LAT_MIN) & (lat <= NAE_LAT_MAX)
    lon_pm180 = np.where(lon > 180, lon - 360, lon)
    lon_mask = (lon_pm180 >= NAE_LON_MIN) & (lon_pm180 <= NAE_LON_MAX)
    return lat_mask, lon_mask


def bounds_metrics_rows(x, coords, config_name, member_id, step, lead_time_hours_val):
    """Per-variable bounds metrics for one (config, member, step) - same
    schema as pipeline/ensemble/02_validate.py's
    build_standalone_variable_table (min/max/violating_fraction), applied
    to the FULL global state (all 73 variables, not just
    STORED_VARIABLES) so a violation is caught regardless of which
    variable it first appears in. Stays on-GPU except the final scalar
    .item() calls. Does not gate anything - see module docstring for why
    this is metrics-only, not a during-inference stop condition."""
    var_names = list(coords["variable"])
    var_axis = list(coords.keys()).index("variable")
    n_total = x.numel() // len(var_names)

    rows = []
    for i, name in enumerate(var_names):
        bounds = VAR_BOUNDS.get(name)
        values = x.select(var_axis, i)
        finite = torch.isfinite(values)
        if bounds is None:
            bad = ~finite
        else:
            lo, hi = bounds
            bad = ~finite | (values < lo) | (values > hi)
        n_bad = int(bad.sum().item())
        finite_values = values[finite]
        rows.append({
            "config": config_name,
            "member": member_id,
            "step": step,
            "lead_time_hours": lead_time_hours_val,
            "variable": name,
            "min_value": finite_values.min().item() if finite_values.numel() else float("nan"),
            "max_value": finite_values.max().item() if finite_values.numel() else float("nan"),
            "n_bad_points": n_bad,
            "n_total_points": n_total,
            "violating_fraction": n_bad / n_total,
        })
    return rows


def cross_variable_metrics_rows(x, coords, config_name, member_id, step, lead_time_hours_val):
    """Cross-variable (z-level ordering) consistency for one (config,
    member, step) - same check and schema as pipeline/ensemble/
    02_validate.py's build_cross_variable_consistency_table: for each
    pair of adjacent geopotential levels, the higher-altitude (lower-
    pressure) level must have the larger z value almost everywhere. This
    is a genuinely different failure mode than bounds_metrics_rows above:
    a field can stay within its own physical bounds at every level while
    the levels' *relative order* still inverts (e.g. z500 dips below
    z700), which a per-variable bounds check alone can't see."""
    var_names = list(coords["variable"])
    var_axis = list(coords.keys()).index("variable")

    rows = []
    for p_low, p_high in zip(Z_LEVELS, Z_LEVELS[1:]):
        var_low, var_high = f"z{p_low}", f"z{p_high}"
        lower = x.select(var_axis, var_names.index(var_low))
        higher = x.select(var_axis, var_names.index(var_high))
        n_total = lower.numel()
        n_violating = int((lower <= higher).sum().item())
        rows.append({
            "config": config_name,
            "member": member_id,
            "step": step,
            "lead_time_hours": lead_time_hours_val,
            "check": f"{var_low}_gt_{var_high}",
            "var_low": var_low,
            "var_high": var_high,
            "n_violating_points": n_violating,
            "n_total_points": n_total,
            "violating_fraction": n_violating / n_total,
        })
    return rows


def _area_weighted_mean(field, lat_axis, lon_axis, lat_weight):
    """Area-weighted (cos-latitude) global mean of a single (..., lat,
    lon, ...) field - same weighting convention as
    e2s.validation.area_weights/global_mean (weight varies only by lat,
    normalized so weight.mean() == 1), reimplemented directly on torch
    tensors to avoid an xarray round-trip on every step."""
    zonal = field.mean(dim=lon_axis)
    w_shape = [1] * zonal.ndim
    w_shape[lat_axis] = -1
    weighted = (zonal * lat_weight.reshape(w_shape)).sum(dim=lat_axis) / lat_weight.sum()
    return weighted.item()


# Global-mean scalars tracked every step - same quantities
# pipeline/ensemble/02_validate.py's cross-time checks use (STEP_JUMP_LIMITS'
# t2m/msl/tcwv/z500, mass_conservation's msl/sp, energy_blowup's kinetic
# energy from u10m/v10m), computed incrementally here instead of by
# reloading the full field afterward.
GLOBAL_MEAN_SIMPLE_VARS = ["t2m", "msl", "sp", "tcwv", "z500"]


def global_mean_metrics_row(x, coords, config_name, member_id, step, lead_time_hours_val):
    """One row of area-weighted global-mean scalars for this (config,
    member, step) - cheap (a handful of numbers), so kept for every step
    of every member rather than needing the full field data to
    reconstruct these later."""
    keys = list(coords.keys())
    var_axis, lat_axis, lon_axis = keys.index("variable"), keys.index("lat"), keys.index("lon")
    var_names = list(coords["variable"])
    lat = torch.as_tensor(coords["lat"], device=x.device, dtype=x.dtype)
    w = torch.cos(torch.deg2rad(lat))
    w = w / w.mean()
    # Selecting out var_axis (< lat_axis/lon_axis in this project's fixed
    # coords ordering) shifts lat/lon down by exactly one each - see
    # select_stored's docstring for the same ordering assumption.
    field_lat_axis, field_lon_axis = lat_axis - 1, lon_axis - 1

    row = {
        "config": config_name, "member": member_id, "step": step,
        "lead_time_hours": lead_time_hours_val,
    }
    for name in GLOBAL_MEAN_SIMPLE_VARS:
        if name in var_names:
            field = x.select(var_axis, var_names.index(name))
            row[f"global_mean_{name}"] = _area_weighted_mean(field, field_lat_axis, field_lon_axis, w)

    if "u10m" in var_names and "v10m" in var_names:
        u = x.select(var_axis, var_names.index("u10m"))
        v = x.select(var_axis, var_names.index("v10m"))
        ke_field = 0.5 * (u**2 + v**2)
        row["global_mean_kinetic_energy"] = _area_weighted_mean(ke_field, field_lat_axis, field_lon_axis, w)

    return row


def select_stored(x, coords, lat_mask, lon_mask):
    """Subset x/coords down to STORED_VARIABLES, cropped to the NAE
    region - the only thing that actually gets written to disk. Relies
    on earth2studio's convention that a coords dict's key order matches
    x's dim order exactly (see e.g. the existing per-step rollout this
    was adapted from, which relies on the same ordering via
    total_coords.move_to_end calls)."""
    var_names = list(coords["variable"])
    var_idx = [var_names.index(v) for v in STORED_VARIABLES]
    keys = list(coords.keys())
    var_axis, lat_axis, lon_axis = keys.index("variable"), keys.index("lat"), keys.index("lon")

    x = x.index_select(var_axis, torch.tensor(var_idx, device=x.device))
    lat_idx = torch.tensor(np.nonzero(lat_mask)[0], device=x.device)
    lon_idx = torch.tensor(np.nonzero(lon_mask)[0], device=x.device)
    x = x.index_select(lat_axis, lat_idx).index_select(lon_axis, lon_idx)

    coords = dict(coords)
    coords["variable"] = np.array(STORED_VARIABLES)
    coords["lat"] = coords["lat"][lat_mask]
    coords["lon"] = coords["lon"][lon_mask]
    return x, coords


def run_config(name, cfg_builder):
    zarr_path = paths.perturbation_zarr_path(name)
    if zarr_path.exists():
        print(f"[SKIP] {name}: {zarr_path} already exists.")
        return
    start_date = START_DATE_OVERRIDES.get(name, START_DATE)
    n_ensemble = N_ENSEMBLE_OVERRIDES.get(name, N_ENSEMBLE)
    print(f"\n=== {name} (start_date={start_date}, n_ensemble={n_ensemble}) ===")
    io = ZarrBackend(str(zarr_path))

    prognostic_ic = model.input_coords()
    time = to_time_array([start_date])
    x0, coords0 = fetch_data(
        source=data, time=time, variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"], device=device,
    )
    cfg_perturbation = cfg_builder(x0, coords0)

    native_coords = model.output_coords(model.input_coords())
    lat_mask, lon_mask = nae_crop_masks(native_coords["lat"], native_coords["lon"])
    cropped_lat = native_coords["lat"][lat_mask]
    cropped_lon = native_coords["lon"][lon_mask]

    lead_time_out = np.asarray(
        [model.output_coords(model.input_coords())["lead_time"] * i for i in range(N_STEPS + 1)]
    ).flatten()

    total_coords = {
        "ensemble": np.arange(n_ensemble),
        "time": time,
        "lead_time": lead_time_out,
        "lat": cropped_lat,
        "lon": cropped_lon,
    }
    io.add_array(total_coords, STORED_VARIABLES)

    lead_time_hours_out = (lead_time_out / np.timedelta64(1, "h")).astype(float)
    bounds_rows = []
    cross_variable_rows = []
    timeseries_rows = []

    for member_id in range(n_ensemble):
        perturbation = Zero() if member_id == 0 else cfg_perturbation
        tag = "control (Zero)" if member_id == 0 else name
        print(f"  member {member_id} [{tag}] ...")

        x = x0.clone().unsqueeze(0)
        coords = {"ensemble": np.array([member_id])} | coords0.copy()
        x, coords = map_coords(x, coords, prognostic_ic)
        x, coords = perturbation(x, coords)  # single IC-only perturbation, mirrors run.ensemble()

        bounds_rows += bounds_metrics_rows(x, coords, name, member_id, 0, lead_time_hours_out[0])
        cross_variable_rows += cross_variable_metrics_rows(x, coords, name, member_id, 0, lead_time_hours_out[0])
        timeseries_rows.append(global_mean_metrics_row(x, coords, name, member_id, 0, lead_time_hours_out[0]))
        x_store, coords_store = select_stored(x, coords, lat_mask, lon_mask)
        io.write(*split_coords(x_store, coords_store))

        for step in range(1, N_STEPS + 1):
            x, coords = model(x, coords)  # one autoregressive step, always taken regardless of prior violations
            rows = bounds_metrics_rows(x, coords, name, member_id, step, lead_time_hours_out[step])
            bounds_rows += rows
            cross_variable_rows += cross_variable_metrics_rows(x, coords, name, member_id, step, lead_time_hours_out[step])
            timeseries_rows.append(global_mean_metrics_row(x, coords, name, member_id, step, lead_time_hours_out[step]))
            worst = max(rows, key=lambda r: r["violating_fraction"])
            if worst["violating_fraction"] > 0:
                print(f"    step {step} (lead_time={lead_time_hours_out[step]:.0f}h): "
                      f"worst variable '{worst['variable']}' violating_fraction={worst['violating_fraction']:.4f}")
            x_store, coords_store = select_stored(x, coords, lat_mask, lon_mask)
            io.write(*split_coords(x_store, coords_store))

    output_path = paths.perturbation_metrics_path
    output_path.mkdir(parents=True, exist_ok=True)
    bounds_path = output_path / f"{name}_bounds.csv"
    cross_variable_path = output_path / f"{name}_cross_variable.csv"
    timeseries_path = output_path / f"{name}_timeseries.csv"
    pd.DataFrame(bounds_rows).to_csv(bounds_path, index=False)
    pd.DataFrame(cross_variable_rows).to_csv(cross_variable_path, index=False)
    pd.DataFrame(timeseries_rows).to_csv(timeseries_path, index=False)
    print(f"Saved '{name}' field data to {zarr_path}, bounds metrics to {bounds_path}, "
          f"cross-variable metrics to {cross_variable_path}, global-mean timeseries to {timeseries_path}")


for config_name, cfg_builder in CONFIGS.items():
    run_config(config_name, cfg_builder)

print("\nAll perturbation configurations complete.")
