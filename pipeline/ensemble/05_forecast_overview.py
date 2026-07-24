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
# # Ensemble forecast: what you get
#
# This chapter shows the raw output of a single FCN3 ensemble run: 8
# members, initialized from the same GFS analysis, perturbed only by the
# model's own stochasticity (`Zero` perturbation - no explicit IC noise).
#
# It starts with a single member's rollout - what one forecast path looks
# like on its own, before any talk of ensembles - then builds up to the
# full picture: every member's path individually, and finally how much the
# members disagree with each other (spread), animated over the rollout.
#
# Validation of these checks (are the members physically plausible?) is a
# separate chapter - see `06_validation_report`.

# %%
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from e2s.paths import ProjPaths
from e2s.validation import drop_time, lead_time_hours, nearest_point

paths = ProjPaths()
paths.ensure_directories()

MUNICH_LAT, MUNICH_LON = 48.1372, 11.5755
COLOR_MEMBER = "#6E7B8B"
COLOR_MEAN = "#1F5C99"

ds = xr.open_zarr(paths.ensemble_zarr_path)
x_hours = lead_time_hours(ds)


def copy_gif(src, dst):
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"[WARN] {src} not found - run 04_analyse.py first.")


# %% [markdown]
# ## One path, animated
#
# A single member's rollout, Robinson projection: 2m temperature, 10m wind
# speed, and 500 hPa geopotential (z500). All 8 members' animations (all
# three fields) are rendered by `04_analyse.py` into `data/ensemble/gifs/`
# - regenerable, not tracked in git. These are copied into `output/` as the
# book's representative example; see `e2s/paths.py`'s `ensemble_gifs_path`
# docstring for why the rest stay out of git.

# %%
copy_gif(
    paths.ensemble_gifs_path / "member_00" / "t2m_robinson.gif",
    paths.ensemble_book_path / "t2m_robinson_member00.gif",
)
copy_gif(
    paths.ensemble_gifs_path / "member_00" / "wind10m_robinson.gif",
    paths.ensemble_book_path / "wind10m_robinson_member00.gif",
)
copy_gif(
    paths.ensemble_gifs_path / "member_00" / "z500_robinson.gif",
    paths.ensemble_book_path / "z500_robinson_member00.gif",
)

# %% [markdown]
# ```{figure} ../../output/ensemble/book/t2m_robinson_member00.gif
# :name: fig-ensemble-t2m-gif
# Member 0's 2m temperature field over the forecast rollout.
# ```
#
# A second member, directly below, for comparison - same field, same
# rollout, different member. Placed right next to each other this way, the
# two animations make member-to-member disagreement visible at a glance,
# before the full ensemble (all 8 members) is even introduced below.
#
# ```{figure} ../../output/ensemble/book/t2m_robinson_member01.gif
# :name: fig-ensemble-t2m-gif-member01
# Member 1's 2m temperature field over the same rollout - compare to
# member 0's animation directly above.
# ```
#
# ```{figure} ../../output/ensemble/book/wind10m_robinson_member00.gif
# :name: fig-ensemble-wind-gif
# Member 0's 10m wind speed field over the forecast rollout.
# ```
#
# ```{figure} ../../output/ensemble/book/z500_robinson_member00.gif
# :name: fig-ensemble-z500-gif
# Member 0's 500 hPa geopotential (z500) field over the forecast rollout.
# ```

# %%
copy_gif(
    paths.ensemble_gifs_path / "member_01" / "t2m_robinson.gif",
    paths.ensemble_book_path / "t2m_robinson_member01.gif",
)

# %% [markdown]
# ## Ensemble
#
# ### Every member's path, individually
#
# Each line is one ensemble member's 2m temperature at Munich over the
# rollout. Unlike a boxplot, this shows whether spread comes from a few
# outlier members or a broad, even spread across all of them.

# %%
munich = nearest_point(ds, MUNICH_LAT, MUNICH_LON)
t2m_munich = drop_time(munich["t2m"]).transpose("lead_time", "ensemble").compute().values - 273.15

fig, ax = plt.subplots(figsize=(14, 5))
for i in range(t2m_munich.shape[1]):
    ax.plot(x_hours, t2m_munich[:, i], color=COLOR_MEMBER, alpha=0.6, linewidth=1.0)
ax.plot(x_hours, t2m_munich.mean(axis=1), color=COLOR_MEAN, linewidth=2.5, label="Ensemble mean")
ax.set_xlabel("Lead time (hours since UTC init)")
ax.set_ylabel("Temperature (deg C)")
ax.set_title("Munich - 2m temperature, every ensemble member")
ax.legend(loc="best")
ax.grid(True, color="#DDDDDD", linewidth=0.6)
fig.tight_layout()
fig.savefig(paths.ensemble_book_path / "spaghetti_t2m_munich.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/ensemble/book/spaghetti_t2m_munich.png
# :name: fig-ensemble-spaghetti-t2m
# Munich 2m temperature, one line per ensemble member.
# ```
#
# ### The same data, summarized
#
# A boxplot per lead-time step compresses the eight lines above into a
# distribution - easier to scan for a long rollout, at the cost of hiding
# which specific member is where. Generated by `04_analyse.py`.
#
# ```{figure} ../../output/ensemble/analysis/meteogram_t2m_munich.png
# :name: fig-ensemble-meteogram-t2m
# Munich 2m temperature ensemble meteogram (boxplot per lead-time step).
# ```
#
# ### Ensemble spread, animated
#
# Instead of comparing individual members by eye, this collapses all 8
# members at each grid point into their standard deviation - a single
# animated field showing where and when the ensemble disagrees with
# itself. Rendered by `04_analyse.py` from the full field, not just
# Munich, so patterns of spread (e.g. higher over land than over open
# ocean) are visible spatially, not just as one time series.

# %%
copy_gif(
    paths.ensemble_gifs_path / "ensemble_std" / "t2m_std_robinson.gif",
    paths.ensemble_book_path / "t2m_std_robinson.gif",
)

# %% [markdown]
# ```{figure} ../../output/ensemble/book/t2m_std_robinson.gif
# :name: fig-ensemble-t2m-std-gif
# Ensemble standard deviation of 2m temperature, over the forecast rollout.
# ```
