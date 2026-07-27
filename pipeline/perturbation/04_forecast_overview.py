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
# # SFNO perturbation sweep: does it tolerate IC noise?
#
# SFNO is a deterministic model - without a perturbation applied to its
# initial condition, every ensemble member is bit-identical, so there's
# no ensemble to speak of. The previous chapter established that SFNO's
# own unperturbed rollout is reliably valid across the year; this one
# asks the next question: once an actual perturbation is added to the
# initial condition, does SFNO tolerate it, and at what amplitude?
#
# It runs SFNO under eight configurations - `zero` (no perturbation, the
# deterministic baseline), `brown_*` (`Brown` noise applied to all 73
# variables, three intensities), `z500_brown_*` (the same `Brown` noise,
# but confined to `z500` alone, same three intensities), and
# `bred_vector` (grows the perturbation via the model itself rather than
# injecting a static offset) - each with one always-`Zero()` control
# member (member 0) to anchor comparisons against, and validates every
# step against this project's physical-plausibility bounds table
# (`e2s.validation.bounds_for`, the same one `pipeline/ensemble/
# 02_validate.py` runs against FCN3) rather than only inspecting the
# final result. Each config also gets its own dedicated chapter (next in
# this section) with a full diagnostic breakdown: which members are
# affected, which variables, whether it's immediate or gradual, and
# whether it's just absolute physical bounds or the *relative*
# consistency between variables (and energy/mass conservation) that
# breaks down.
#
# One design change from an earlier version of this sweep, and the
# single biggest factor in the results below: `brown_*` and
# `z500_brown_*`'s noise amplitude is now calibrated *per variable*,
# proportional to that variable's own spatial standard deviation in the
# initial condition, rather than one flat number applied identically to
# all 73 variables (see `pipeline/perturbation/01_run.py`'s
# `ScaledBrownPerturbation` and `compute_variable_scales`). A flat
# `noise_amplitude=0.05` is negligible for `z500` (spatial std on the
# order of 1e3 m^2/s^2) and enormous for `u10m` (spatial std of a few
# m/s) - the same nominal number was two completely different
# experiments depending on which variable it landed on, which is exactly
# why an earlier version of this sweep saw `brown_0.05` blow up
# catastrophically. Gaussian noise is dropped from this sweep entirely:
# it failed for the identical flat-amplitude reason, and IID-per-
# gridpoint noise has no spatial reddening structure to calibrate the
# same way `Brown` does.
#
# See `pipeline/perturbation/01_run.py`'s module docstring for the full
# methodology, including why the check is metrics-only rather than a
# stop-on-violation gate: SFNO's raw `tcwv` output is mildly negative
# across ~7% of the global grid even fully unperturbed, unlike FCN3
# (which passes this exact bounds table with zero violations) - a
# single-bit gate can't distinguish that baseline idiosyncrasy from
# genuine divergence.

# %%
import pandas as pd
import xarray as xr

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

CONFIG_NAMES = [
    "zero", "brown_0.05", "brown_0.01", "brown_0.002",
    "z500_brown_0.05", "z500_brown_0.01", "z500_brown_0.002", "bred_vector",
]

# %% [markdown]
# ## Summary: worst-variable violating fraction and ensemble spread at 120h
#
# For each config: the fraction of the global grid still outside its
# physical bounds (worst variable) at the final step (120h), control
# (member 0, always `Zero()`) vs. the mean across the other, actually-
# perturbed members - plus the final-step ensemble spread (std across
# members) of Munich-region `t2m`, to separate "did it stay physically
# valid" from "did it actually produce a usable ensemble spread."

# %%
rows = []
for name in CONFIG_NAMES:
    bounds_path = paths.perturbation_metrics_path / f"{name}_bounds.csv"
    if not bounds_path.exists():
        print(f"[WARN] {bounds_path} not found - run 01_run.py first.")
        continue
    df = pd.read_csv(bounds_path)
    last_step = df["step"].max()
    worst = df[df["step"] == last_step].groupby("member")["violating_fraction"].max()

    ds = xr.open_zarr(paths.perturbation_zarr_path(name))
    t2m_final = ds["t2m"].isel(lead_time=-1).values
    spread = float(t2m_final.std(axis=0).mean())

    rows.append({
        "config": name,
        "control (member 0)": worst.get(0, float("nan")),
        "perturbed members (mean)": worst.drop(index=0, errors="ignore").mean(),
        "t2m ensemble spread, K (120h)": spread,
    })

summary = pd.DataFrame(rows).set_index("config")
print(summary.round(4))

# %% [markdown]
# The control column sits at the same ~0.054 baseline in every row - the
# same deterministic SFNO forecast every time, regardless of config. The
# headline result: with per-variable calibration, `brown_*` and
# `z500_brown_*` all sit at that same ~0.054 baseline too, at every
# intensity tested - statistically indistinguishable from `zero`. That's
# a direct reversal from the flat-amplitude version of this sweep, where
# `brown_0.05` reached >90% violating fraction within the first step.
# The spread column confirms these aren't just "too weak to do
# anything" - `t2m` ensemble spread scales cleanly with intensity, from
# 0.03 K (`brown_0.002`) up to 0.56 K (`brown_0.05`), and z500-only
# perturbation produces real (if smaller) spread too, without the
# validity cost. `bred_vector` is the outlier in both directions: its
# perturbed-member violating fraction (~0.49) is elevated - but, as that
# chapter details, confined almost entirely to two stratospheric-
# humidity variables already sitting at noise-floor scale, not genuine
# divergence - and it produces by far the largest ensemble spread (8.3 K),
# an order of magnitude beyond any `brown_*` intensity tested here.
#
# The next eight chapters give each config its own full breakdown -
# per-member, per-variable, cross-variable ordering, and energy/mass
# conservation over the rollout.
