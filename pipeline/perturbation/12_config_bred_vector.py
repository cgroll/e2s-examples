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
# # `bred_vector`: the largest-spread method, with one caveat
#
# Every other config in this section (`brown_*`, `z500_brown_*`) adds a
# static offset in raw physical units, calibrated to each variable's own
# scale. `BredVector` doesn't do that at all: it grows the perturbation
# by running SFNO itself for 20 internal steps on a lightly-`Brown`-
# seeded start (seeded at amplitude 0.002), then rescales the *result* to
# a target amplitude relative to the state's own norm (`gamma = norm(x) /
# norm(x + dx)` in earth2studio's implementation) - see
# `pipeline/perturbation/01_run.py`'s module docstring for the exact
# mechanics. It was the first method in this project's perturbation work
# that produced a clean result, before the `brown_*`/`z500_brown_*`
# per-variable scaling fix below also turned out to work.
#
# **What's actually happening:** this is close to what a working ensemble
# perturbation should look like. `t2m` is at exactly 0% grid-point
# violation at *every* step, for every member - not just low, zero. The
# z-level ordering check (panel C) is flat at 0.0 throughout, identical
# to the control. Kinetic energy stays in a 0.7-1.5x band around step 0
# (final ratios: 0.85x, 0.99x, 1.42x across the three perturbed members) -
# genuine member-to-member variation, not runaway growth, and nowhere
# close to `pipeline/ensemble/02_validate.py`'s 5x FCN3 threshold. Mass
# drift stays under 0.3%. And this is by far the largest ensemble spread
# in the section: Munich `t2m` spread reaches 8.3 K by the final step
# and `z500` spread reaches 1836 m^2/s^2 - roughly 15x `brown_0.05`'s
# spread (the largest of the calibrated-noise configs) at a comparable
# level of physical validity.
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
# 0.44-0.49, not climbing toward 1.0. (B) the violation is concentrated
# almost entirely in `q50`/`q100` - everything else reads clean green,
# same as every other config in this section. (C) z-level ordering - flat
# zero, identical to the control. (D) kinetic energy ratio - stays in a
# healthy band around 1.0, the widest member-to-member spread in this
# section. (E) mass drift - under 0.3% throughout.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The largest spread in this section - expect visibly more ensemble
# variation here than in any `brown_*`/`z500_brown_*` chapter, while
# still looking like a physically coherent forecast rather than a field
# losing structure.

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
# looking meteogram here - real ensemble spread around the control,
# visibly wider than any `brown_*`/`z500_brown_*` config, not members
# diverging off-scale.

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
# standalone-variable heatmap should show two clearly red rows
# (`q50`/`q100`) against an otherwise green field, and the cross-variable
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
# Both approaches in this section now produce physically valid ensembles:
# `bred_vector`, and the per-variable-scaled `brown_*`/`z500_brown_*`
# family. The difference is spread magnitude and structure - `bred_vector`
# produces roughly an order of magnitude more spread than any calibrated
# `Brown` intensity tested, by perturbing the full state in a flow-
# consistent way rather than a fixed set of variables at a fixed
# amplitude, at the cost of the `q50`/`q100` caveat above. The one open
# item is the same as before: either accept `q50`/`q100` as a known,
# provably-benign artifact of a strict `>= 0` bound meeting an
# already-near-zero physical quantity, or loosen those two variables'
# bounds slightly for stratospheric humidity specifically. Carrying a
# working method back into `pipeline/downscaling/01_run.py` in place of
# `Brown(noise_amplitude=0.05)` is still a separate, not-yet-made change
# to a different experiment's pipeline.
