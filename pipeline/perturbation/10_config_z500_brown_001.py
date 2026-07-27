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
# # `z500_brown_0.01`: `z500`-only scaled noise, intensity 0.01
#
# Same as the previous chapter - `ScaledBrownPerturbation` confined to
# `z500` alone via `SingleVariablePerturbation` - at intensity 0.01
# instead of 0.05, matching the middle `brown_0.01` intensity from the
# all-variable family.
#
# **What's actually happening:** the same clean result, with
# proportionally less spread. Worst-variable violating fraction is
# 0.0692 at step 1 and 0.0538 at the final step (120h) - identical to
# `zero`. Z-level ordering is exactly 0.0 throughout, kinetic energy
# stays at 0.797-0.799x step 0, mass drift is effectively zero. Munich
# `t2m` spread reaches 0.04 K by the final step (vs. 0.21 K for
# `z500_brown_0.05`) and `z500` spread reaches 13 m^2/s^2 (vs. 72) -
# intensity scales spread roughly linearly here too, same as the
# all-variable family.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: z500_brown_0.01 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.01_dashboard.png
# :name: fig-perturbation-z500brown001-dashboard
# Statistically indistinguishable from `zero`'s dashboard - no elevated
# violating fraction, no z-level ordering breakdown, kinetic energy and
# mass drift both within the clean range.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Visibly smaller spread than `z500_brown_0.05`'s - the same fields, at
# roughly a fifth the intensity.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.01_mean_std.gif
# :name: fig-perturbation-z500brown001-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.01_meteogram_t2m_munich.png
# :name: fig-perturbation-z500brown001-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.01_meteogram_wind10m_munich.png
# :name: fig-perturbation-z500brown001-wind10m
# Munich 10m wind speed - same layout.
# ```
