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
# # `z500_brown_200`: single-variable, scale-appropriate noise
#
# A direct test of a hypothesis from the earlier configs: `Brown`/
# `Gaussian` at `noise_amplitude=0.05` were applied identically across
# all 73 variables at once, but 0.05 is wildly different scale relative
# to different variables - negligible for `z500` (~55,000 m^2/s^2),
# catastrophic for e.g. `u10m` (~1.3 m/s). This config perturbs *only*
# `z500`, with `Brown` noise at `noise_amplitude=200` - still small
# relative to `z500`'s own scale (about 0.36%), but no longer negligible
# the way 0.05 was. Every other one of the 73 variables is left at its
# exact, unperturbed IC value (see `pipeline/perturbation/01_run.py`'s
# `SingleVariablePerturbation` wrapper).
#
# **What's actually happening:** nothing pathological, at any point in
# the rollout. `t2m` and z-level ordering are at exactly 0% violation for
# every member at every step - not just low, identical to the clean
# `zero` baseline. Kinetic energy stays in a 0.79-0.81x band relative to
# step 0 (matching `zero`'s own ~0.8x dissipation almost exactly), and
# mass drift is effectively zero (-0.01%). The Munich meteogram below
# shows real, growing ensemble spread - visibly wider boxes at later lead
# times - tracking the same synoptic trajectory as the control, which is
# what a physically sensible ensemble perturbation should look like. This
# is, together with `bred_vector`, the second config in this sweep that
# doesn't diverge - and unlike `bred_vector`, it doesn't carry the
# `q50`/`q100` stratospheric-humidity side effect either, at least at
# this amplitude.
#
# The obvious caveat: this only perturbs one variable. It's a much
# narrower ensemble-generation strategy than `bred_vector` (which
# perturbs the full state, just in a flow-consistent way) - useful as a
# clean confirmation that *properly scaled* noise is tolerable, less
# useful as a drop-in ensemble method on its own unless perturbing z500
# alone turns out to be enough to represent the forecast uncertainty
# that matters for a given use case.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: z500_brown_200 - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_200_dashboard.png
# :name: fig-perturbation-z500brown200-dashboard
# All five panels are statistically indistinguishable from the clean
# `zero` baseline's dashboard - no elevated violating fraction anywhere,
# no z-level ordering breakdown, kinetic energy and mass drift both
# within the same range as the unperturbed control.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Unlike every static-noise config that perturbed all 73 variables, this
# should show a visible but modest ensemble std - real spread, not
# either "exactly zero" (`zero`) or "everything's lost physical
# structure" (`brown_0.05` etc.).

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_200_mean_std.gif
# :name: fig-perturbation-z500brown200-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/z500_brown_200_meteogram_t2m_munich.png
# :name: fig-perturbation-z500brown200-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot). Real, growing spread, tracking the control's
# trajectory.
# ```
#
# ```{figure} ../../output/perturbation/analysis/z500_brown_200_meteogram_wind10m_munich.png
# :name: fig-perturbation-z500brown200-wind10m
# Munich 10m wind speed - same layout. Notably, wind speed shows visible
# spread too even though only `z500` was perturbed directly - consistent
# with the model coupling a z500 perturbation into other fields through
# its own dynamics, rather than the spread being an isolated artifact of
# the one perturbed channel.
# ```
