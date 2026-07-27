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
# # SFNO perturbation sweep: does it tolerate IC noise?
#
# SFNO is a deterministic model - without a perturbation applied to its
# initial condition, every ensemble member is bit-identical, so there's
# no ensemble to speak of. This chapter asks the more basic question
# first, before picking a perturbation strategy: does SFNO even tolerate
# noise added to its initial condition, and at what amplitude?
#
# It runs SFNO under five configurations - `zero` (no perturbation, the
# deterministic baseline), and `Brown`/`Gaussian` noise at several
# amplitudes - each with one always-`Zero()` control member (member 0) to
# anchor comparisons against, and validates every step against this
# project's physical-plausibility bounds table (`e2s.validation.
# bounds_for`, the same one `pipeline/ensemble/02_validate.py` runs
# against FCN3) rather than only inspecting the final result. Each
# config also gets its own dedicated chapter (next in this section) with
# a full diagnostic breakdown: which members are affected, which
# variables, whether it's immediate or gradual, and whether it's just
# absolute physical bounds or the *relative* consistency between
# variables (and energy/mass conservation) that breaks down.
#
# See `pipeline/perturbation/01_run.py`'s module docstring for the full
# methodology, including why the check is metrics-only rather than a
# stop-on-violation gate: SFNO's raw `tcwv` output is mildly negative
# across ~7% of the global grid even fully unperturbed, unlike FCN3
# (which passes this exact bounds table with zero violations) - a
# single-bit gate can't distinguish that baseline idiosyncrasy from
# genuine divergence.

# %%
import pandas as pd

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

CONFIG_NAMES = ["zero", "brown_0.05", "brown_0.01", "brown_0.002", "gaussian_0.05"]

# %% [markdown]
# ## Summary: worst-variable violating fraction at the end of the rollout
#
# For each config, the fraction of the global grid still outside its
# physical bounds (worst variable) at the final step (120h) - control
# (member 0, always `Zero()`) vs. the mean across the other, actually-
# perturbed members.

# %%
rows = []
for name in CONFIG_NAMES:
    path = paths.perturbation_metrics_path / f"{name}_bounds.csv"
    if not path.exists():
        print(f"[WARN] {path} not found - run 01_run.py first.")
        continue
    df = pd.read_csv(path)
    last_step = df["step"].max()
    worst = df[df["step"] == last_step].groupby("member")["violating_fraction"].max()
    rows.append({
        "config": name,
        "control (member 0)": worst.get(0, float("nan")),
        "perturbed members (mean)": worst.drop(index=0, errors="ignore").mean(),
    })

summary = pd.DataFrame(rows).set_index("config")
print(summary.round(4))

# %% [markdown]
# The control column should sit at roughly the same ~0.05-0.07 baseline
# in every row regardless of config - it's the same deterministic SFNO
# forecast every time. The perturbed-member column is the real signal:
# every perturbed config here ends up far higher than that baseline,
# including the smallest amplitude tested (`brown_0.002`, 25x smaller
# than the default `Brown` amplitude of 0.05).
#
# The next five chapters give each config its own full breakdown -
# per-member, per-variable, cross-variable ordering, and energy/mass
# conservation over the rollout.
