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
# # `brown_0.01`: Brown noise, intensity 0.01 (per-variable scaled)
#
# Same per-variable-scaled `Brown` noise as the previous chapter, 5x
# smaller intensity - amplitude = `0.01 * that variable's own spatial
# standard deviation`, applied to all 73 variables.
#
# **What's actually happening:** the same clean result as `brown_0.05`,
# just with proportionally less ensemble spread. Worst-variable violating
# fraction is 0.0694 at step 1 and 0.0541 at the final step (120h) -
# statistically identical to `zero`'s 0.0692 / 0.0542, and to
# `brown_0.05`'s own numbers. Z-level ordering is exactly 0.0 throughout.
# Kinetic energy stays in a 0.797-0.799x band relative to step 0 (the
# same dissipation as every other clean config in this section), and
# mass drift is effectively zero (-0.01%). The only real difference from
# `brown_0.05` is the ensemble's *magnitude*: Munich `t2m` spread reaches
# 0.15 K by the final step (vs. 0.56 K for `brown_0.05`) and `z500`
# spread reaches 39 m^2/s^2 (vs. 160) - intensity scales spread roughly
# linearly, as expected from the calibration, without changing validity
# at all.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: brown_0.01 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.01_dashboard.png
# :name: fig-perturbation-brown001-dashboard
# Statistically indistinguishable from `zero`'s and `brown_0.05`'s
# dashboards - no elevated violating fraction, no z-level ordering
# breakdown, kinetic energy and mass drift both within the clean range.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The std panels here should show real spread, visibly smaller than
# `brown_0.05`'s - the same fields, at roughly a fifth the intensity.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.01_mean_std.gif
# :name: fig-perturbation-brown001-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.01_meteogram_t2m_munich.png
# :name: fig-perturbation-brown001-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). Real spread, smaller than `brown_0.05`'s.
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.01_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown001-wind10m
# Munich 10m wind speed - same layout.
# ```
