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
# # `zero`: no perturbation (baseline)
#
# SFNO has no internal stochasticity, so with `Zero()` every member is
# bit-identical - there's no ensemble here, just four copies of the same
# deterministic forecast. This is the reference case every other config
# in this section gets compared against.
#
# **What's actually happening:** nothing pathological. At step 1, 0 of
# 73 variables exceed 50% grid-point violation (mean across the "perturbed"
# members, which for this config are just more `Zero()` copies) - the
# worst offenders are the same mildly-negative-`tcwv` idiosyncrasy
# described in `pipeline/perturbation/01_run.py`'s docstring (~5-7% of
# the grid, present even here), not genuine divergence. The z-level
# ordering check (panel C) is perfectly clean at every step - exactly
# 0.0 violating fraction throughout. Kinetic energy actually *decreases*
# slightly relative to step 0 (~0.8x by 120h - ordinary dissipation, not
# growth), and mean sea-level pressure drift is indistinguishable from
# zero. Every number in this config's dashboard is what "nothing wrong"
# looks like - worth keeping in view when reading the other four
# chapters in this section.

# %%
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()
print("Config: zero (no perturbation) - see pipeline/perturbation/01_run.py for the full sweep methodology.")

# %% [markdown]
# ## Diagnostics dashboard

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/zero_dashboard.png
# :name: fig-perturbation-zero-dashboard
# (A) per-member worst-variable bounds-violating fraction - flat at the
# baseline. (B) most-affected-variables heatmap - uniformly pale, none
# exceed the baseline noise floor. (C) z-level ordering - exactly zero
# throughout. (D) kinetic energy ratio - stays below 1.0 (mild
# dissipation). (E) mass drift - indistinguishable from zero.
# ```

# %% [markdown]
# ## Relation to the main ensemble's validation methodology
#
# Panels A/B/C above aren't a new methodology - they're this sweep's
# version of the same two checks `pipeline/ensemble/02_validate.py` runs
# for the main FCN3 ensemble (see the "Ensemble forecast" section's
# validation report chapter), computed per (config, member, step,
# variable) instead of per (member, lead_time, variable) since there's a
# sweep of configs here rather than one run. The two full heatmaps below
# are the direct equivalent of that report's
# `standalone_variable_summary_heatmap.png` and
# `cross_variable_consistency_summary_heatmap.png` - same layout (rows =
# variable/check, columns = lead time, color = fraction violating),
# built from this config's own data. Panels A and B/C in the dashboard
# above are both *derived* from tables shaped like these: panel A takes,
# at each step, the single worst-scoring row (collapsed across all 73
# variables) and plots it as one line per member instead of averaging
# across members into one aggregate row per cell; panel B is the same
# heatmap as below, just cropped to the ~25 highest-scoring rows so it
# stays legible at dashboard scale; panel C does the same collapse-to-
# worst-row-per-step as panel A, but starting from the cross-variable
# table instead of the standalone one.
#
# For `zero`, both heatmaps should be uniformly pale (standalone) or
# exactly blank (cross-variable) - nothing here exceeds the baseline.
# Compare directly against the same two figures in e.g. `brown_0.05`'s
# chapter, where the standalone heatmap turns solid dark red almost
# immediately and the cross-variable heatmap lights up across most
# adjacent z-level pairs.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/zero_standalone_heatmap.png
# :name: fig-perturbation-zero-standalone-heatmap
# Standalone variable bounds, all 73 variables - mean violating fraction
# across the perturbed members (member 0, the control, excluded), per
# variable and lead time.
# ```
#
# ```{figure} ../../output/perturbation/analysis/zero_cross_variable_heatmap.png
# :name: fig-perturbation-zero-cross-variable-heatmap
# Cross-variable (z-level ordering) consistency, all 11 adjacent-level
# checks - same layout.
# ```

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# The standard deviation panels below should render as flat, uniform
# color - zero spread, since every member is identical.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/zero_mean_std.gif
# :name: fig-perturbation-zero-gif
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region.
# ```

# %% [markdown]
# ## Munich meteograms
#
# With every member identical, the boxplots collapse to a flat line -
# there's no box to draw, only the control-member marker.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/zero_meteogram_t2m_munich.png
# :name: fig-perturbation-zero-t2m
# Munich 2m temperature.
# ```
#
# ```{figure} ../../output/perturbation/analysis/zero_meteogram_wind10m_munich.png
# :name: fig-perturbation-zero-wind10m
# Munich 10m wind speed.
# ```
