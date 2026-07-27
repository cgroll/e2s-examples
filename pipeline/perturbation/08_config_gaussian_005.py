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
# # `gaussian_0.05`: IID Gaussian noise, amplitude 0.05
#
# IID `Gaussian` noise on the initial condition, amplitude 0.05 - same
# amplitude as `brown_0.05`, but spatially uncorrelated (no spectral
# reddening) rather than correlated. This is the one config in the
# sweep that isolates *noise structure* (correlated vs. white) rather
# than amplitude.
#
# **What's actually happening:** this is the fastest and most uniform
# failure of the five configs. `tcwv` is already at 100% grid-point
# violation by step 1 (vs. `brown_0.05`'s 99%) and 70 of 73 variables
# exceed 50% violation at that same first step - matching `brown_0.05`'s
# breadth, reached slightly faster. Unlike the Brown configs, `t2m`
# itself is directly in the top-5 affected variables from step 1 (along
# with `q1000`, `q600`, `q700`) - white noise appears to hit temperature
# more directly than the spatially-smoothed Brown noise does at the same
# amplitude. The z-level ordering check (~47-52%) and kinetic energy
# growth (71,450-71,512x step 0) land in the same range as `brown_0.05`,
# but with a genuinely different character: the three perturbed
# members' kinetic-energy trajectories sit within a few hundred of each
# other throughout (compare `brown_0.05`'s 49,000-107,000x spread, or
# `brown_0.002`'s 73x-923x spread) - white noise seems to produce a
# more member-independent failure mode than correlated noise does, at
# least at this amplitude. Mass drift (-85% to -86%) is similarly tight
# across members.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: gaussian_0.05 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/gaussian_0.05_dashboard.png
# :name: fig-perturbation-gaussian005-dashboard
# (A) all perturbed members saturate within the first step or two. (B)
# `t2m` appears directly among the most-affected variables, unlike the
# Brown configs. (C)
# z-level ordering violations, comparable to `brown_0.05`. (D) kinetic
# energy ratio, log scale - note the tight clustering of the three
# member curves compared to the Brown configs' dashboards. (E) mass
# drift, similarly tight across members.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/gaussian_0.05_mean_std.gif
# :name: fig-perturbation-gaussian005-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# With `t2m` itself among the most-affected variables from step 1
# onward, expect the perturbed members to separate from the control
# immediately here too - similar to `brown_0.05`, not `brown_0.002`.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/gaussian_0.05_meteogram_t2m_munich.png
# :name: fig-perturbation-gaussian005-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/gaussian_0.05_meteogram_wind10m_munich.png
# :name: fig-perturbation-gaussian005-wind10m
# Munich 10m wind speed - same layout.
# ```
