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
# # `bred_vector`: a method that actually works
#
# Every static-noise config in this section (`brown_0.05`, `brown_0.01`,
# `brown_0.002`, `gaussian_0.05`) diverged, regardless of amplitude - the
# common thread was adding one flat noise amplitude in raw physical units
# across all 73 variables at once, which is enormous relative to some
# variables' natural scale and negligible relative to others'. `BredVector`
# doesn't do that: it grows the perturbation by running SFNO itself for 20
# internal steps on a lightly-`Brown`-seeded start (seeded at amplitude
# 0.002, the least-bad amplitude from the earlier sweep - the seed still
# has to survive those 20 steps before it's used), then rescales the
# *result* to a target amplitude relative to the state's own norm
# (`gamma = norm(x) / norm(x + dx)` in earth2studio's implementation) -
# see `pipeline/perturbation/01_run.py`'s module docstring for the exact
# mechanics.
#
# **What's actually happening:** this is close to what a working ensemble
# perturbation should look like. `t2m` is at exactly 0% grid-point
# violation at *every* step, for every member - not just low, zero. The
# z-level ordering check (panel C) is flat at 0.0 throughout, identical
# to the control. Kinetic energy stays in a 0.7-1.5x band around step 0
# (final ratios: 0.85x, 0.99x, 1.42x across the three perturbed members) -
# genuine member-to-member variation, not runaway growth, and nowhere
# close to `pipeline/ensemble/02_validate.py`'s 5x FCN3 threshold. Mass
# drift stays under 0.3% (vs. the four failed configs' 14-100%). This is
# the first config in this sweep whose kinetic-energy and mass panels
# actually look like a normal, physically plausible ensemble instead of a
# blow-up in progress.
#
# It isn't perfectly clean, though: panel B shows the violation is
# concentrated almost entirely in `q50` and `q100` (specific humidity at
# the two highest, driest levels), sitting around 40-49% and essentially
# flat across the whole rollout - not spreading to other variables, not
# growing. Checking the actual values explains why: `q50`'s min/max
# across the rollout is roughly &plusmn;0.00008 kg/kg - noise-floor scale,
# about 625x smaller than the bound table's own 0.05 kg/kg ceiling. The
# stratosphere is essentially dry, so specific humidity there sits right
# at the edge of the bound's strict `>= 0` requirement, and even a faint
# bred-vector perturbation is enough to push a large fraction of those
# already-near-zero grid points across it - the same category of
# harmless bounds-table artifact as the `tcwv`/baseline idiosyncrasy
# described in `01_run.py`'s docstring, just concentrated in different
# variables here instead of spread thinly across many.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: bred_vector - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/bred_vector_dashboard.png
# :name: fig-perturbation-bredvector-dashboard
# (A) per-member worst-variable violating fraction - stable around
# 0.47-0.51, not climbing toward 1.0 like every static-noise config. (B)
# the violation is concentrated almost entirely in `q50`/`q100` -
# everything else is pale. (C) z-level ordering - flat zero, identical to
# the control. (D) kinetic energy ratio - stays in a healthy band around
# 1.0. (E) mass drift - under 0.3% throughout.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Unlike the failed configs, expect this to look like an actual weather
# forecast throughout the rollout, not a field losing physical structure.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/bred_vector_mean_std.gif
# :name: fig-perturbation-bredvector-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# With `t2m` at 0% violation throughout, expect a genuinely normal-
# looking meteogram here - real, modest ensemble spread around the
# control, not members diverging off-scale.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/bred_vector_meteogram_t2m_munich.png
# :name: fig-perturbation-bredvector-t2m
# Munich 2m temperature - control member (orange) vs. the perturbed
# members (grey boxplot).
# ```
#
# ```{figure} ../../output/perturbation/analysis/bred_vector_meteogram_wind10m_munich.png
# :name: fig-perturbation-bredvector-wind10m
# Munich 10m wind speed - same layout.
# ```

# %% [markdown]
# ## Full standalone-variable and cross-variable heatmaps
#
# Same layout as the `zero` chapter's reference figures - here, the
# standalone-variable heatmap should show two clearly dark rows
# (`q50`/`q100`) against an otherwise pale field, and the cross-variable
# heatmap should be blank throughout.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/bred_vector_standalone_heatmap.png
# :name: fig-perturbation-bredvector-standalone-heatmap
# Standalone variable bounds, all 73 variables.
# ```
#
# ```{figure} ../../output/perturbation/analysis/bred_vector_cross_variable_heatmap.png
# :name: fig-perturbation-bredvector-cross-variable-heatmap
# Cross-variable (z-level ordering) consistency, all 11 adjacent-level
# checks.
# ```

# %% [markdown]
# ## Where this leaves the sweep
#
# `bred_vector` is the first (and so far only) config here that doesn't
# diverge. The one open item is `q50`/`q100`: either accept it as a
# known, provably-benign artifact of a strict `>= 0` bound meeting an
# already-near-zero physical quantity (consistent with how this project
# already treats `tcwv`'s baseline idiosyncrasy), or loosen those two
# variables' bounds slightly for stratospheric humidity specifically.
# Either way, this is now the candidate to actually carry back into
# `pipeline/downscaling/01_run.py` in place of `Brown(noise_amplitude=0.05)`
# - not yet done here, since that's a separate change to a different
# experiment's pipeline.
