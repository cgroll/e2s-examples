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
# # `z500_brown_0.002`: `z500`-only scaled noise, intensity 0.002
#
# The smallest intensity in the `z500_brown_*` family - `z500`-only
# `ScaledBrownPerturbation` at intensity 0.002, matching `brown_0.002`
# from the all-variable family.
#
# **What's actually happening:** the same clean result as every other
# config in this section, with the smallest ensemble spread of the
# entire sweep. Worst-variable violating fraction is 0.0692 at step 1
# and 0.0542 at the final step (120h) - indistinguishable from `zero` to
# four decimal places. Z-level ordering is exactly 0.0 throughout,
# kinetic energy stays at 0.799x step 0 for every member (the tightest
# clustering of any config in this section), mass drift is effectively
# zero. Munich `t2m` spread reaches just 0.007 K by the final step and
# `z500` spread reaches 1.7 m^2/s^2 - real, but barely so; at this
# intensity the perturbation is close to the smallest one that still
# produces a measurable ensemble at all.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: z500_brown_0.002 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.002_dashboard.png
# :name: fig-perturbation-z500brown0002-dashboard
# Statistically indistinguishable from `zero`'s dashboard - no elevated
# violating fraction, no z-level ordering breakdown, and the tightest
# cross-member kinetic-energy clustering of any config in this section.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The smallest spread in this entire sweep - expect the std panels here
# to look very close to `zero`'s flat, blank ones.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.002_mean_std.gif
# :name: fig-perturbation-z500brown0002-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.002_meteogram_t2m_munich.png
# :name: fig-perturbation-z500brown0002-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). The smallest spread of any config in this
# section.
# ```
#
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.002_meteogram_wind10m_munich.png
# :name: fig-perturbation-z500brown0002-wind10m
# Munich 10m wind speed - same layout.
# ```
