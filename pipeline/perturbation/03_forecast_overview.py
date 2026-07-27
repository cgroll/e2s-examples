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
# The "Temporal downscaling" chapter's SFNO+InterpModAFNO forecast uses
# `Brown(noise_amplitude=0.05)` to perturb the initial condition - without
# it, SFNO is deterministic and every ensemble member would be identical
# (see that chapter's meteograms, which collapse to zero-height boxes
# under `Zero()`). But that combination turned out to blow up: `t2m`,
# `z500`, `u10m`, `v10m` all diverge to unphysical magnitudes from the
# very first native SFNO step.
#
# This chapter isolates the question: is that InterpModAFNO's fault, or
# does bare SFNO itself not tolerate this kind of perturbation? It runs
# SFNO directly (no interpolation wrapper) under five configurations -
# `zero` (no perturbation, the deterministic baseline), and `Brown`/
# `Gaussian` noise at several amplitudes - each with one always-`Zero()`
# control member (member 0) to anchor comparisons against, and validates
# every step against this project's physical-plausibility bounds table
# (`e2s.validation.bounds_for`, the same one `pipeline/ensemble/
# 02_validate.py` runs against FCN3) rather than only inspecting the
# final result.
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
# forecast every time. The perturbed-member column is the real signal.
#
# ```{figure} ../../output/perturbation/analysis/blowup_summary.png
# :name: fig-perturbation-blowup-summary
# Worst-variable bounds-violating fraction over the full 120h rollout,
# one panel per config, one line per member. Flat near the baseline means
# no genuine divergence; a sharp rise toward 1.0 means the member's state
# left the physically plausible range and stayed there.
# ```
#
# Every perturbed config diverges within about 20 hours - including
# `brown_0.002`, forty times smaller than earth2studio's own default
# amplitude (0.05). The control member (orange) stays flat at the
# baseline in every panel, confirming the divergence is driven by the
# perturbation itself, not an artifact of the validation check.
#
# ## Munich meteograms: baseline vs. the smallest amplitude tested
#
# Not `brown_0.05` (the config already known to fail) - `brown_0.002`,
# forty times smaller, to make the point that this isn't primarily an
# amplitude-calibration problem.
#
# ```{figure} ../../output/perturbation/analysis/zero_meteogram_t2m_munich.png
# :name: fig-perturbation-zero-meteogram
# `zero`: every member is the identical deterministic SFNO forecast, so
# the boxplot collapses to a flat line - the expected, unperturbed
# baseline.
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.002_meteogram_t2m_munich.png
# :name: fig-perturbation-brown002-meteogram
# `brown_0.002`: the perturbed members (grey) have already left the
# physically plausible temperature range within the first day, while the
# control member (orange) tracks the same baseline forecast as the `zero`
# config above.
# ```
#
# ## Spatial view: ensemble mean/std over the rollout
#
# The same contrast, spatially, over the North-Atlantic-European region.
#
# ```{figure} ../../output/perturbation/analysis/zero_mean_std.gif
# :name: fig-perturbation-zero-gif
# `zero`: ensemble mean tracks a physically plausible forecast throughout;
# ensemble std is exactly zero everywhere (no perturbation, deterministic
# model).
# ```
#
# ```{figure} ../../output/perturbation/analysis/brown_0.05_mean_std.gif
# :name: fig-perturbation-brown05-gif
# `brown_0.05`: both mean and std fields lose physical meaning within the
# first few frames as the perturbed members diverge.
# ```
#
# ## Interpretation and next step
#
# This doesn't look like a pure amplitude-calibration problem - scaling
# `Brown`'s amplitude down by 40x still fails. The more likely culprit:
# `Brown`/`Gaussian` here add spatially-uniform noise, at one fixed
# amplitude, directly onto SFNO's raw physical-unit state across all 73
# variables at once. Those variables span wildly different physical
# scales (`z500` ~56,000 vs. `u10m` ~1.3), so a single amplitude that's
# negligible for one variable can be enormous relative to another -
# there's no per-variable calibration happening at all.
#
# Not yet tried here: bred-vector methods (`BredVector`,
# `HemisphericCentredBredVector`), which grow the perturbation
# dynamically via short pre-rollout integration of the model itself
# rather than injecting static external noise - by construction, the
# result stays flow-consistent rather than an arbitrary offset in raw
# physical units. That's the natural next step before revisiting the
# temporal downscaling chapter's perturbation choice.
