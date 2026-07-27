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
# # Does bare SFNO produce valid forecasts, regardless of season?
#
# Before picking a perturbation strategy at all, this chapter asks a more
# basic question: is SFNO's own *unperturbed* rollout reliably valid
# across the year, or was every other result in this section (and the
# "Population-weighted temperature" chapter this section grew out of)
# built on top of one date's idiosyncrasy?
#
# Four single (not ensemble) deterministic `Zero()`-perturbation
# forecasts, one initial condition per season - 2026-01-15 (winter),
# 2026-04-15 (spring), 2026-07-23 (summer, the same date every other
# config in this section uses), 2025-10-15 (autumn). "Single" is
# deliberate: SFNO has no internal stochasticity, so a `Zero()`-perturbed
# ensemble of any size would just be N bit-identical copies of the same
# forecast - the four ensemble members in every other config in this
# section only start diverging once an actual perturbation is applied.
# Running one member per date instead of four is the same experiment for
# a quarter of the GPU cost, and it's a cleaner statement of the actual
# question: does *the control member* - the reference every other
# config in this section is validated against - stay valid regardless of
# which initial condition it started from?

# %%
import pandas as pd

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

CONFIG_DATES = {
    "ic_robustness_winter": "2026-01-15",
    "ic_robustness_spring": "2026-04-15",
    "ic_robustness_summer": "2026-07-23",
    "ic_robustness_autumn": "2025-10-15",
}

# %% [markdown]
# ## Summary: worst-variable and tcwv violating fraction, per season

# %%
rows = []
for name, date in CONFIG_DATES.items():
    df = pd.read_csv(paths.perturbation_metrics_path / f"{name}_bounds.csv")
    last_step = df["step"].max()
    tcwv = df[df["variable"] == "tcwv"].set_index("step")["violating_fraction"]
    worst = df.groupby("step")["violating_fraction"].max()
    rows.append({
        "config": name, "date": date,
        "worst-variable step 1": worst.loc[1],
        "worst-variable final (120h)": worst.loc[last_step],
        "tcwv step 1": tcwv.loc[1],
        "tcwv final (120h)": tcwv.loc[last_step],
    })

summary = pd.DataFrame(rows).set_index("config")
print(summary.round(4))

# %% [markdown]
# All four dates stay well-behaved for the full 5-day rollout: the
# worst-variable violating fraction never exceeds ~7% of the global grid
# at any step, in any season, and in three of the four cases it actually
# *decreases* over the rollout rather than growing (winter: 2.6% -> 1.5%;
# spring: 6.8% -> 2.3%; autumn: 5.8% -> 2.6%; only summer stays roughly
# flat, 6.9% -> 5.4%). There's no case here that resembles the genuine
# divergence this project has seen elsewhere (`pipeline/downscaling`'s
# SFNO+InterpModAFNO+`Brown(0.05)` combination, or this section's own
# flat-amplitude Brown/Gaussian configs before they were recalibrated -
# see the next chapter) - every one of these four numbers is the same
# kind of small, stable, moisture-tracking idiosyncrasy documented
# throughout this section, not a step toward blow-up.
#
# In every season, the worst offender is a humidity variable (`tcwv` or
# an upper-air `q`-level), and `tcwv` itself tracks the season directly:
# lowest in winter (2.6% -> 1.5%), highest in summer (6.9% -> 5.4%), with
# spring and autumn in between. That's consistent with a physical
# explanation rather than a numerical one - the grid has genuinely less
# near-zero moisture to dip infinitesimally negative against the bounds
# table's strict `>= 0` requirement when the atmosphere itself is drier.
# A genuine model instability would have no obvious reason to track the
# season's actual moisture content this closely.

# %% [markdown]
# ```{figure} ../../output/perturbation/analysis/ic_robustness_comparison.png
# :name: fig-perturbation-ic-robustness-comparison
# (A) worst-variable bounds-violating fraction, one line per season -
# all four stay in the same low, non-growing range. (B) the same
# comparison isolated to `tcwv` - the seasonal ordering (winter lowest,
# summer highest) is visible directly.
# ```

# %% [markdown]
# ## What this does and doesn't establish
#
# This confirms the *deterministic control* is trustworthy across the
# year - the reference line every dashboard in this section plots as
# "control (member 0)" isn't a summer-specific artifact. It does not,
# on its own, say anything about how well a *perturbed* ensemble behaves
# in a different season; the next chapters build up which perturbation
# strategies keep an actual ensemble within bounds, all still anchored to
# the 2026-07-23 date this control-robustness check has now validated as
# a reasonable one to build on.
