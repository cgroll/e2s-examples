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
# # `brown_0.05`: Brown noise, intensity 0.05 (per-variable scaled)
#
# Spatially correlated (`Brown`) noise added to the initial condition of
# all 73 variables, amplitude = `0.05 * that variable's own spatial
# standard deviation` (see `pipeline/perturbation/01_run.py`'s
# `ScaledBrownPerturbation`) - not a flat `noise_amplitude=0.05` applied
# identically everywhere, which is what earlier versions of this sweep
# used and what `pipeline/downscaling/01_run.py` still uses. 0.05 is the
# largest of the three intensities tested in this section.
#
# **What's actually happening:** nothing pathological, at any point in
# the rollout. The worst-variable violating fraction is 0.0690 at step 1
# and 0.0539 at the final step (120h) - statistically identical to the
# unperturbed `zero` baseline (0.0692 / 0.0542), not the >90%,
# near-instant saturation this same nominal amplitude produced before
# calibration. Z-level ordering (panel C) is exactly 0.0 throughout, at
# every step for every member. Kinetic energy stays in a 0.78-0.82x band
# relative to step 0 across all four members - matching `zero`'s own
# dissipation almost exactly - and mass drift is effectively zero
# (-0.01%). Unlike the flat-amplitude version, this is a real,
# meaningfully-sized ensemble: Munich `t2m` spread reaches 0.56 K by the
# final step, and `z500` spread reaches 160 m^2/s^2 - the largest spread
# of the three `brown_*` intensities, and comparable in physical scale to
# `z500_brown_0.05`'s single-variable version (this section's overview
# chapter compares all eight configs directly).

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: brown_0.05 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.05_dashboard.png
# :name: fig-perturbation-brown005-dashboard
# All five panels are statistically indistinguishable from the clean
# `zero` baseline's dashboard - no elevated violating fraction anywhere,
# no z-level ordering breakdown, kinetic energy and mass drift both
# within the same range as the unperturbed control.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Unlike `zero` (flat, zero spread - every member identical), the std
# panels below should show real, visible spread - genuine ensemble
# variation from perturbing all 73 variables, without the loss of
# physical structure the flat-amplitude version of this config showed.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.05_mean_std.gif
# :name: fig-perturbation-brown005-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.05_meteogram_t2m_munich.png
# :name: fig-perturbation-brown005-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). Real, growing spread, tracking the control's
# trajectory.
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.05_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown005-wind10m
# Munich 10m wind speed - same layout.
# ```
