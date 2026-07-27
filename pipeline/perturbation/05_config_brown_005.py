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
# # `brown_0.05`: Brown noise, amplitude 0.05
#
# Spatially correlated (`Brown`) noise added to the initial condition,
# amplitude 0.05 - earth2studio's own default for this perturbation
# class, and the same value `pipeline/downscaling/01_run.py` uses for
# its SFNO+InterpModAFNO ensemble.
#
# **What's actually happening:** this is total, near-instant failure.
# 70 of 73 variables already exceed 50% grid-point violation at the
# very first step (mean across perturbed members) - `tcwv` alone hits
# 99% by step 1. This isn't a slow-building problem: the divergence is
# already almost complete before the rollout is 6 hours in, and stays
# saturated near 100% for the rest of the 120h window. The z-level
# ordering check (panel C) confirms this isn't just an absolute-bounds
# artifact either - roughly 55-65% of adjacent geopotential-level pairs
# are inverted throughout, meaning the vertical structure of the
# atmosphere itself has stopped making physical sense. Kinetic energy
# reaches 49,000-107,000x its step-0 value (compare against
# `pipeline/ensemble/02_validate.py`'s own growth-factor threshold of
# 5x for FCN3), and mean sea-level pressure has drifted 85-100% from its
# starting value by the end of the rollout - both far past anything
# describable as "conserved."

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
# (A) every perturbed member saturates near 1.0 within the first couple
# of steps. (B) the most-affected variables are dominated by moisture
# (`tcwv`, `q1000`, `q850`) but also reach wind and temperature (`u150`,
# `t100`) - this isn't confined to one physical quantity. (C) z-level
# ordering violations plateau around 0.5-0.6, not just absolute bounds.
# (D) kinetic energy ratio, log scale - reaches five orders of magnitude
# above step 0. (E) mass drift reaches roughly -100%.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The mean/std color scales below use a robust (1st/99th percentile)
# range specifically because a handful of exploded grid points would
# otherwise make the entire scale meaningless for every other frame -
# even so, expect the fields to visibly lose physical structure within
# the first few frames.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.05_mean_std.gif
# :name: fig-perturbation-brown005-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# `t2m` alone reaches 95-98% grid-point violation globally by step 1
# (see the per-variable data), so unlike `brown_0.002` (next-but-one
# chapter), there's no "it looks fine locally" gap here - Munich's own
# temperature trace should already show the perturbed members clearly
# off the control's baseline from the first box onward.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.05_meteogram_t2m_munich.png
# :name: fig-perturbation-brown005-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.05_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown005-wind10m
# Munich 10m wind speed - same layout.
# ```
