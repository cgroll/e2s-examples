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
# # `brown_0.002`: Brown noise, amplitude 0.002
#
# Spatially correlated (`Brown`) noise, amplitude 0.002 - 25x smaller
# than earth2studio's default (`brown_0.05`).
#
# **What's actually happening - the most nuanced case in this sweep:**
# at this amplitude, the failure is genuinely concentrated rather than
# global. Only 2 of 73 variables exceed 50% grid-point violation at
# step 1 (vs. 34/73 for `brown_0.01` and 70/73 for `brown_0.05`), and
# the top-5 affected variables at step 1 are *all* moisture -
# `tcwv`, `q925`, `q100`, `q50`, `q1000` - not a single temperature,
# wind, or geopotential variable makes the list. `t2m` itself starts
# mild (15% grid-point violation at step 1) and only climbs to ~58% by
# mid-rollout before easing back to ~53% - a visibly different,
# *gradual* curve compared to the other three perturbed configs' near-
# instant saturation. So a first glance at just the Munich `t2m`
# meteogram here could plausibly look "not that bad."
#
# It isn't, though: by the final step the worst variable still reaches
# 96.5% violation, z-level ordering sits at 35-40% (lower than
# `brown_0.01`/`brown_0.05`'s ~50%, but far from clean), and kinetic
# energy growth is both large *and* strikingly inconsistent across
# members - 73x, 93x, and 923x step 0 respectively. That order-of-
# magnitude spread between members at the *same* amplitude is itself a
# finding: at this amplitude, whether a given random draw ends up mild
# or severe seems to depend on more than just the noise magnitude.
# Mass drift is comparatively small (+1.4%, +0.4%, -9.4%) - the one
# metric here that stays roughly within `pipeline/ensemble/
# 02_validate.py`'s conservation tolerance for two of the three members.
#
# Note on scope: the moisture variables driving this config's failure
# (`tcwv`, `q*`) aren't among this project's stored spatial fields
# (`t2m`/`u10m`/`v10m`/`z500` only, see `pipeline/perturbation/
# 01_run.py`'s `STORED_VARIABLES`) - panel B below is the closest view
# into where the problem actually concentrates; there's no spatial map
# of it here.

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
# (A) perturbed members climb more gradually than the two larger Brown
# amplitudes. (B) the most-affected variables here are moisture-dominated -
# worth comparing directly against `brown_0.01`'s and `brown_0.05`'s
# panel B, where wind/temperature appear much earlier. (C) z-level
# ordering violations, lower than the two larger amplitudes but still
# well above zero. (D) kinetic energy ratio, log scale - note how far
# apart the three perturbed members' curves are, unlike the tighter
# clustering in the other Brown/Gaussian configs. (E) mass drift - the
# smallest-magnitude of the four perturbed configs.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Since `t2m`/wind/`z500` (the fields actually mapped below) are milder
# here than the moisture variables driving panel B, expect this gif to
# look visibly less catastrophic than `brown_0.05`'s or `brown_0.01`'s -
# that's consistent with, not a contradiction of, this config's failure
# being concentrated elsewhere.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.002_mean_std.gif
# :name: fig-perturbation-brown0002-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# This is the one config in this sweep where the `t2m` meteogram alone
# could be misleading: `t2m`'s own violation curve builds up gradually
# (15% at step 1, ~58% by mid-rollout) rather than saturating instantly,
# so the perturbed members may look closer to the control early in the
# rollout than the dashboard's other panels would suggest.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/brown_0.002_meteogram_t2m_munich.png
# :name: fig-perturbation-brown0002-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.002_meteogram_wind10m_munich.png
# :name: fig-perturbation-brown0002-wind10m
# Munich 10m wind speed - same layout.
# ```
