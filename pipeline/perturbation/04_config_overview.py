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
# ## About this chapter
#
# Standardized breakdown, same structure for every perturbation config in
# this section, selected by the `PERTURBATION_CONFIG` environment
# variable this notebook was executed with: (A) is every ensemble member
# affected, and does divergence appear immediately or build up gradually;
# (B) are all variables affected, or only some; (C) does the divergence
# show up only as an absolute-bounds violation, or does the *relative*
# ordering between variables (geopotential height across pressure levels)
# break too; (D)/(E) is energy/mass conserved over the rollout. Member 0
# is always the unperturbed `Zero()` control, highlighted in every panel
# below - it should sit at the same flat baseline in every config's
# dashboard, since it's the same deterministic SFNO forecast regardless
# of what the other members are perturbed with.

# %%
import os

from IPython.display import Markdown, display

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

CONFIG_NAME = os.environ.get("PERTURBATION_CONFIG", "zero")

CONFIG_DESCRIPTIONS = {
    "zero": "No perturbation - the deterministic baseline. SFNO has no "
            "internal stochasticity, so every member is bit-identical.",
    "brown_0.05": "`Brown` (spatially correlated) noise on the initial "
                  "condition, amplitude 0.05 - earth2studio's own default "
                  "amplitude for this perturbation class.",
    "brown_0.01": "`Brown` noise, amplitude 0.01 - 5x smaller than the "
                  "default amplitude.",
    "brown_0.002": "`Brown` noise, amplitude 0.002 - 25x smaller than "
                   "the default amplitude.",
    "gaussian_0.05": "IID `Gaussian` noise on the initial condition, "
                      "amplitude 0.05 - same amplitude as `brown_0.05`, "
                      "but spatially uncorrelated rather than "
                      "spectrally reddened.",
}
description = CONFIG_DESCRIPTIONS.get(CONFIG_NAME, "")

display(Markdown(f"# Perturbation config: `{CONFIG_NAME}`\n\n{description}"))

# %% [markdown]
# ## Diagnostics dashboard
#
# (A) per-member worst-variable bounds-violating fraction over the
# rollout. (B) the 10 most-affected variables (mean across perturbed
# members). (C) z-level ordering violations - a field can stay within its
# own bounds while its order relative to neighboring levels still
# inverts, which (A)/(B) alone can't see. (D) kinetic energy relative to
# step 0, against the same growth-factor threshold
# `pipeline/ensemble/02_validate.py` uses for FCN3. (E) mean sea-level
# pressure drift relative to step 0, against that same script's mass-
# conservation tolerance.

# %%
display(Markdown(
    f"```{{figure}} ../../output/perturbation/analysis/{CONFIG_NAME}_dashboard.png\n"
    f":name: fig-perturbation-{CONFIG_NAME}-dashboard\n"
    f"Diagnostics dashboard for `{CONFIG_NAME}`.\n"
    f"```"
))

# %% [markdown]
# ## Spatial view: ensemble mean and spread
#
# Ensemble mean and standard deviation of 2m temperature, 10m wind speed,
# and 500 hPa geopotential, over the North-Atlantic-European region, one
# frame per lead-time step.

# %%
display(Markdown(
    f"```{{figure}} ../../output/perturbation/analysis/{CONFIG_NAME}_mean_std.gif\n"
    f":name: fig-perturbation-{CONFIG_NAME}-gif\n"
    f"Ensemble mean/std, `{CONFIG_NAME}`.\n"
    f"```"
))

# %% [markdown]
# ## Munich meteograms
#
# Control member (orange) vs. the perturbed members (grey boxplot).

# %%
display(Markdown(
    f"```{{figure}} ../../output/perturbation/analysis/{CONFIG_NAME}_meteogram_t2m_munich.png\n"
    f":name: fig-perturbation-{CONFIG_NAME}-t2m\n"
    f"Munich 2m temperature, `{CONFIG_NAME}`.\n"
    f"```\n\n"
    f"```{{figure}} ../../output/perturbation/analysis/{CONFIG_NAME}_meteogram_wind10m_munich.png\n"
    f":name: fig-perturbation-{CONFIG_NAME}-wind10m\n"
    f"Munich 10m wind speed, `{CONFIG_NAME}`.\n"
    f"```"
))
