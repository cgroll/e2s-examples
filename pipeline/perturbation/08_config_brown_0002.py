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
# # `brown_0.002`: Brown noise, intensity 0.002 (per-variable scaled)
#
# Same per-variable-scaled `Brown` noise as the previous two chapters,
# 25x smaller than `brown_0.05` - amplitude = `0.002 * that variable's
# own spatial standard deviation`, applied to all 73 variables. The
# smallest of the three intensities tested in this section.
#
# **What's actually happening:** the same clean result as `brown_0.05`
# and `brown_0.01`, scaled down further. Worst-variable violating
# fraction is 0.0692 at step 1 and 0.0542 at the final step (120h) -
# indistinguishable from `zero` to four decimal places. Z-level ordering
# is exactly 0.0 throughout, kinetic energy stays at 0.799-0.800x step 0
# (the tightest clustering across members of any `brown_*` intensity),
# and mass drift is effectively zero (-0.01%). The ensemble spread this
# produces is correspondingly small - Munich `t2m` spread reaches only
# 0.03 K by the final step, `z500` spread reaches 6.5 m^2/s^2 - real but
# modest, the smallest of the three `brown_*` intensities. Where an
# earlier (flat-amplitude) version of this sweep found 0.002 to be the
# amplitude where Brown's failure mode shifted from "broad and immediate"
# to "concentrated in moisture variables, but still ultimately a
# failure," the calibrated version doesn't have a failure mode to shift
# at any of the three intensities tested - see this section's overview
# chapter for the full before/after comparison.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: brown_0.002 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.002_dashboard.png
# :name: fig-perturbation-brown0002-dashboard
# Statistically indistinguishable from `zero`'s dashboard - no elevated
# violating fraction, no z-level ordering breakdown, and the tightest
# cross-member kinetic-energy/mass-drift clustering of any `brown_*`
# intensity in this section.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The smallest spread of the three `brown_*` intensities - expect the
# std panels here to look close to (but not quite) `zero`'s flat, blank
# ones.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.002_mean_std.gif
# :name: fig-perturbation-brown0002-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.002_meteogram_t2m_munich.png
# :name: fig-perturbation-brown0002-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). The smallest spread of the three `brown_*`
# intensities.
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.002_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown0002-wind10m
# Munich 10m wind speed - same layout.
# ```
