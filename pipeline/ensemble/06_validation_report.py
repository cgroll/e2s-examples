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
# # Ensemble validation: did it actually work?
#
# A forecast can run to completion and still be physically nonsense - NaNs,
# blown-up kinetic energy, a collapsed ensemble spread. `02_validate.py`
# checks for that across four independent dimensions, each producing one
# table (`output/ensemble/validation/tables/*.csv`); `03_validate_visualize.py`
# renders them. This chapter is the narrated summary of those results, not a
# re-run of the checks themselves.
#
# | Check | Grain | What it catches |
# |---|---|---|
# | Standalone variable bounds | (variable, member, lead_time) | NaN/Inf, physically impossible values (e.g. negative humidity) |
# | Cross-variable consistency | (check, member, lead_time) | Violated physical ordering between variables (e.g. geopotential must decrease with pressure level) |
# | Cross-time consistency | (check, member, transition) | Step jumps, mass-conservation drift, kinetic-energy blow-up between consecutive steps |
# | Cross-ensemble consistency | (check, lead_time) | Ensemble spread collapsing to zero or exploding |

# %%
from pathlib import Path

import pandas as pd

from e2s.paths import ProjPaths

paths = ProjPaths()
tables_dir = paths.ensemble_validation_tables_path

tables = {
    "standalone_variable": pd.read_csv(tables_dir / "standalone_variable.csv"),
    "cross_variable_consistency": pd.read_csv(tables_dir / "cross_variable_consistency.csv"),
    "cross_time_consistency": pd.read_csv(tables_dir / "cross_time_consistency.csv"),
    "cross_ensemble_consistency": pd.read_csv(tables_dir / "cross_ensemble_consistency.csv"),
}

summary = pd.DataFrame([
    {
        "check": name,
        "rows": len(df),
        "invalid": int((~df["valid"]).sum()) if not df.empty and "valid" in df else 0,
        "status": "SKIPPED (empty)" if df.empty else ("PASS" if (~df["valid"]).sum() == 0 else "FAIL"),
    }
    for name, df in tables.items()
])
summary

# %% [markdown]
# ## Standalone variable bounds
#
# Every (variable, member, lead_time) triple checked against physically
# reasonable bounds (`e2s/validation.py`'s `VARIABLE_BOUNDS`) - e.g. 2m
# temperature must be between 150K and 340K. The heatmap below is the
# aggregate across all 8 members; red means a larger fraction of members
# violated bounds at that (variable, lead_time) cell.
#
# ```{figure} ../../output/ensemble/validation/standalone_variable_summary_heatmap.png
# :name: fig-validation-standalone
# Fraction of members with out-of-bounds values, per variable and lead time.
# ```

# %% [markdown]
# ## Cross-time consistency
#
# Three transition-based checks per rollout: how much a variable's
# area-weighted global mean can jump between consecutive steps
# (`STEP_JUMP_LIMITS`), whether mass (mean sea level pressure) drifts too
# far from step 0, and whether 10m kinetic energy blows up relative to step
# 0. Each flagged point marks *both* endpoints of an invalid transition,
# since the jump could originate from either side.
#
# ```{figure} ../../output/ensemble/validation/cross_time_step_jump_t2m.png
# :name: fig-validation-step-jump-t2m
# Step-to-step 2m temperature jumps against the configured limit.
# ```
#
# ```{figure} ../../output/ensemble/validation/cross_time_energy_blowup_kinetic_energy.png
# :name: fig-validation-ke-blowup
# 10m kinetic energy relative to its step-0 value.
# ```

# %% [markdown]
# ## Cross-ensemble consistency
#
# Std-dev across members of the global-mean field, tracked over the
# rollout. Collapsing toward zero means the members converged onto the same
# trajectory (not physically expected for a stochastic model); exploding
# means something diverged. Compared against the spread at the first
# lead-time where it becomes nonzero, not lead-time 0 - see the docstring
# above `build_cross_ensemble_consistency_table` in `02_validate.py` for why
# lead-time 0 is a degenerate baseline with an unperturbed IC.
#
# ```{figure} ../../output/ensemble/validation/cross_ensemble_ensemble_spread_t2m.png
# :name: fig-validation-ensemble-spread
# Ensemble spread of 2m temperature over the rollout.
# ```

# %% [markdown]
# ## Cross-variable consistency
#
# Checks physical ordering between pressure-level variables: geopotential
# must increase with altitude, i.e. decrease with pressure, so e.g.
# `z500 > z700 > z850` should hold almost everywhere. `01_run.py` writes
# FCN3's full 72-variable state (all pressure levels) rather than a
# subset, so this check has adjacent-level pairs to compare across the
# whole atmospheric column.
#
# ```{figure} ../../output/ensemble/validation/cross_variable_consistency_summary_heatmap.png
# :name: fig-validation-cross-variable
# Fraction of members violating pressure-level ordering, per check and lead time.
# ```
