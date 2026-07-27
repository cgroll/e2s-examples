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
# # `brown_0.01`: Brown noise, amplitude 0.01
#
# Spatially correlated (`Brown`) noise, amplitude 0.01 - 5x smaller than
# earth2studio's default (`brown_0.05`, previous chapter).
#
# **What's actually happening:** still a full failure, just a somewhat
# less immediate one. 34 of 73 variables exceed 50% grid-point violation
# at step 1 (vs. 70/73 for `brown_0.05`) - roughly half the initial
# breadth - and the worst step-1 offender is `u150` (upper-level wind),
# not `tcwv`. But "less immediate" only buys a few hours: by the final
# step, 99.5% of the worst variable's grid points are still violating,
# essentially indistinguishable from `brown_0.05`'s end state. The
# z-level ordering check sits around 0.46-0.53 - close to `brown_0.05`'s
# range, meaning the vertical-structure breakdown isn't meaningfully
# gentler even though the absolute-bounds onset is slower. Kinetic
# energy reaches 3,200-8,400x step 0 (an order of magnitude less than
# `brown_0.05`'s 49,000-107,000x, but still nowhere near
# `pipeline/ensemble/02_validate.py`'s 5x FCN3 threshold), and mass
# drift reaches -14% to -37% - smaller than `brown_0.05`, but still far
# outside anything resembling conservation.
#
# The pattern across `brown_0.05` -> `brown_0.01` -> `brown_0.002`
# (next chapter) is a genuine dose-response relationship in *severity*
# and *onset speed*, but not a threshold below which the problem
# disappears - see this section's overview chapter for the interpretation.

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
# (A) perturbed members climb to saturation over the first ~10 steps,
# slightly slower than `brown_0.05`. (B) wind (`u150`) and moisture
# (`tcwv`, `q850`, `q1000`) lead, temperature (`t100`) follows. (C)
# z-level ordering violations, comparable magnitude to `brown_0.05`. (D)
# kinetic energy ratio, log scale. (E) mass drift.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.01_mean_std.gif
# :name: fig-perturbation-brown001-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# `t2m` reaches ~78-90% grid-point violation globally within the first
# few steps here (less immediate than `brown_0.05`'s ~95%, but still
# high) - expect the perturbed members to separate from the control
# fairly quickly, if not quite as abruptly as in the previous chapter.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.01_meteogram_t2m_munich.png
# :name: fig-perturbation-brown001-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.01_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown001-wind10m
# Munich 10m wind speed - same layout.
# ```
