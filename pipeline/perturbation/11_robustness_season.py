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
# # Robustness check: does the season matter?
#
# Every config so far used the same initial condition date (2026-07-23,
# northern-hemisphere summer). Before trusting any of this sweep's
# findings as a general property of SFNO rather than an artifact of one
# particular atmospheric state, the two most load-bearing results - the
# clean `zero` baseline and the working `bred_vector` method - are rerun
# from a northern-hemisphere winter initial condition (2026-01-15)
# instead, with everything else (model, ensemble size, rollout length,
# perturbation parameters) unchanged.

# %%
import pandas as pd

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()


def summarize(name):
    df = pd.read_csv(paths.perturbation_metrics_path / f"{name}_bounds.csv")
    last_step = df["step"].max()
    control = df[df["member"] == 0]
    perturbed = df[df["member"] != 0]
    return {
        "config": name,
        "control step 1": control[control["step"] == 1]["violating_fraction"].max(),
        "control final": control[control["step"] == last_step]["violating_fraction"].max(),
        "perturbed step 1 (mean)": perturbed[perturbed["step"] == 1].groupby("member")["violating_fraction"].max().mean() if not perturbed.empty else float("nan"),
        "perturbed final (mean)": perturbed[perturbed["step"] == last_step].groupby("member")["violating_fraction"].max().mean() if not perturbed.empty else float("nan"),
    }


summary = pd.DataFrame([summarize(n) for n in ["zero", "zero_winter", "bred_vector", "bred_vector_winter"]]).set_index("config")
print(summary.round(4))

# %% [markdown]
# ## What replicates
#
# The `zero` baseline's `tcwv` idiosyncrasy (documented throughout this
# section) shrinks in winter - 6.9% at step 1 in summer vs. 2.6% in
# winter, 5.4% vs. 1.5% by the end of the rollout - but doesn't disappear,
# and the control-member behavior is otherwise identical (deterministic,
# flat, no z-level or energy/mass issues in either season - not shown
# here, see each config's own bounds/cross_variable/timeseries CSVs).
# That's consistent with the interpretation already used for this
# artifact: total column water vapor is genuinely lower in winter, so
# there's less near-zero moisture available to dip infinitesimally
# negative against the bounds table's strict `>= 0` requirement. This is
# evidence *for* that interpretation, not just an assumption - a
# genuine numerical instability would have no obvious reason to track
# the season's actual moisture content this closely.
#
# `bred_vector`'s signature reproduces closely across both seasons:
# 49.1% (summer) vs. 45.0% (winter) at step 1, 48.7% vs. 44.1% by the
# final step - the same stable, non-growing, `q50`/`q100`-concentrated
# pattern described in that config's own chapter, not a summer-specific
# fluke. That's a meaningfully stronger claim than the single-date result
# alone: whatever is happening with bred_vector and stratospheric
# humidity is a property of the method interacting with SFNO, not an
# artifact of one particular atmospheric state.
#
# ## Scope of this check
#
# This reruns only the two configs whose results this sweep's overall
# conclusions actually depend on - not the full six-plus-config sweep a
# second time. The static-noise configs (`brown_*`, `gaussian_0.05`)
# were not rerun under a second date: nothing about *why* they diverge
# (a flat amplitude applied across variables of wildly different
# physical scale) is date-dependent, so a second date would be very
# unlikely to change that conclusion and wasn't worth the additional GPU
# time to confirm.
