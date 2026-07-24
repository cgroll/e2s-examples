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
# # Does writing fewer variables per step actually help?
#
# `01_run.py` (the ensemble baseline) already trims its zarr output to a
# handful of variables via `output_coords={"variable": [...]}}`. Reading
# `earth2studio.run.ensemble`'s source showed why that's not free compute
# savings: `prognostic.create_iterator()` always yields the model's *full*
# internal state every step - FCN3 needs the whole state to roll forward -
# and `output_coords` only filters immediately before `io.write()`. So the
# claim is: subsetting should cut **storage and write time**, but not
# **model compute time**.
#
# `01_compare_full_vs_subset.py` checks that empirically: the same small
# FCN3 ensemble, run twice, once with the full variable set and once with a
# 4-variable subset.

# %%
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from e2s.paths import ProjPaths

paths = ProjPaths()
df = pd.read_csv(paths.variable_subset_output_path / "full_vs_subset.csv").set_index("run")
df

# %%
time_ratio = df.loc["subset", "wall_time_s"] / df.loc["full", "wall_time_s"]
size_ratio = df.loc["subset", "size_mb"] / df.loc["full", "size_mb"]
print(f"Time ratio (subset/full):    {time_ratio:.3f}")
print(f"Storage ratio (subset/full): {size_ratio:.3f}")
if time_ratio < 0.8:
    print(
        "-> Wall-clock time dropped too, not just storage. Model compute per "
        "step is unchanged (see 01_compare_full_vs_subset.py's docstring for "
        "why) - so for this run size, zarr write I/O was a large enough "
        "fraction of total time that cutting it sped up the whole run."
    )
else:
    print("-> Wall-clock time barely moved: here, I/O was a small fraction of total time, so only storage improved.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].bar(df.index, df["wall_time_s"], color=["#6E7B8B", "#1F5C99"])
axes[0].set_ylabel("Wall time (s)")
axes[0].set_title("Compute time")
axes[1].bar(df.index, df["size_mb"], color=["#6E7B8B", "#1F5C99"])
axes[1].set_ylabel("Zarr size (MB)")
axes[1].set_title("Storage")
fig.tight_layout()
fig.savefig(paths.variable_subset_output_path / "full_vs_subset_bars.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/variable_subset/full_vs_subset_bars.png
# :name: fig-variable-subset-bars
# Full-variable-set vs. 4-variable-subset runs: compute time (left) vs.
# on-disk size (right).
# ```
#
# Reading `earth2studio.run.ensemble`'s source shows the model always
# computes its full internal state every step regardless of
# `output_coords` - so subsetting can't reduce GPU compute. Whether it also
# reduces *wall-clock* time depends on how much of that time is spent on
# zarr I/O for the run size in question: for this short benchmark
# (`nsteps`/`nensemble` in the table above), I/O turned out to be a large
# enough share of total time that the measured time ratio printed above
# isn't close to 1.0 - subsetting sped up the whole run, not just the
# storage footprint. For a longer rollout, where I/O is a smaller fraction
# of total time, expect that ratio to drift closer to 1.0.
