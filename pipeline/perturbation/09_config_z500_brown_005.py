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
# # `z500_brown_0.05`: same scaled noise, confined to `z500` alone
#
# The `brown_*` family perturbs all 73 variables at once, each at
# `intensity * that variable's own spatial std`. This config applies the
# identical `ScaledBrownPerturbation` and the identical intensity (0.05)
# to *only* `z500`, via `SingleVariablePerturbation` - every other
# variable is left at its exact, unperturbed IC value (see
# `pipeline/perturbation/01_run.py`). The question this isolates: does
# confining the perturbation to one variable change anything about
# validity, or just about how much of the state actually varies across
# members?
#
# **What's actually happening:** the same clean result as `brown_0.05`,
# with less ensemble spread. Worst-variable violating fraction is 0.0691
# at step 1 and 0.0539 at the final step (120h) - identical to `zero`
# and to the all-variable `brown_0.05`. Z-level ordering is exactly 0.0
# throughout, kinetic energy stays at 0.79-0.80x step 0, and mass drift
# is effectively zero. The one real difference from `brown_0.05` is
# spread magnitude: Munich `t2m` spread reaches 0.21 K by the final step
# (vs. 0.56 K for the all-variable version) and `z500` spread reaches 72
# m^2/s^2 (vs. 160) - roughly a third to a half, consistent with only one
# of 73 variables actually being perturbed. Notably, `t2m` and `wind10m`
# still show real spread despite never being perturbed directly - the
# model's own dynamics couple a `z500` perturbation into other fields
# over the rollout, rather than the spread being confined to the one
# channel that was actually touched.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: z500_brown_0.05 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.05_dashboard.png
# :name: fig-perturbation-z500brown005-dashboard
# Statistically indistinguishable from `zero`'s and `brown_0.05`'s
# dashboards - no elevated violating fraction, no z-level ordering
# breakdown, kinetic energy and mass drift both within the clean range.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Real, visible spread in all three fields - despite only `z500` being
# perturbed directly - though smaller than `brown_0.05`'s all-variable
# version.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.05_mean_std.gif
# :name: fig-perturbation-z500brown005-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.05_meteogram_t2m_munich.png
# :name: fig-perturbation-z500brown005-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). Real, growing spread despite `t2m` never being
# perturbed directly.
# ```
#
# ```{figure} ../../output/perturbation/analysis/z500_brown_0.05_meteogram_wind10m_munich.png
# :name: fig-perturbation-z500brown005-wind10m
# Munich 10m wind speed - same layout.
# ```
