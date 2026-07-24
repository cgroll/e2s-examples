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
# # Temporal downscaling: 6h native steps to 1h
#
# `InterpModAFNO` wraps SFNO (a 6h-native prognostic model) and inserts 5
# AFNO-interpolated steps between each pair of SFNO's own 6h outputs,
# producing an hourly forecast. Crucially, the steps at the 6h boundaries
# aren't touched by the interpolation model - they're SFNO's own output,
# passed straight through. So within the single hourly zarr this experiment
# writes, every 6th lead-time step is a genuine model prediction and
# everything in between is AFNO-synthesized.
#
# This chapter compares the two directly: the same field, the same
# rollout, once at native 6h resolution and once at the interpolated 1h
# resolution - stacked so the added temporal detail (and which of it is
# "real") is visible at a glance.

# %%
import shutil

import matplotlib
matplotlib.use("Agg")

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()


def copy_asset(src, dst):
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"[WARN] {src} not found - run 02_analyse.py first.")


# %% [markdown]
# ## Munich meteogram: dense vs. native-only
#
# The hourly boxplot below has roughly 6x as many x-axis entries as the
# native-only one directly beneath it - same rollout, same member spread,
# just far more temporal detail. The orange boxes in the top panel mark
# the steps that are SFNO's own native 6h output; the grey boxes around
# them are AFNO-interpolated.

# %%
copy_asset(
    paths.downscaling_analysis_path / "meteogram_t2m_munich_dense.png",
    paths.downscaling_book_path / "meteogram_t2m_munich_dense.png",
)
copy_asset(
    paths.downscaling_analysis_path / "meteogram_t2m_munich_native.png",
    paths.downscaling_book_path / "meteogram_t2m_munich_native.png",
)

# %% [markdown]
# ```{figure} ../../output/downscaling/book/meteogram_t2m_munich_dense.png
# :name: fig-downscaling-meteogram-dense
# Munich 2m temperature ensemble meteogram, hourly resolution. Orange boxes
# are native SFNO steps; grey boxes are AFNO-interpolated.
# ```
#
# ```{figure} ../../output/downscaling/book/meteogram_t2m_munich_native.png
# :name: fig-downscaling-meteogram-native
# The same rollout, subsampled to only the native 6h SFNO steps - what
# you'd see without the interpolation model.
# ```
#
# ## One member, animated: dense vs. native-only
#
# Same idea, animated: member 0's 2m temperature field, once with all 25
# hourly frames (each tagged as a native SFNO step or an AFNO-interpolated
# one) and once with only the 5 native frames directly below it. All
# members' gifs (both variants) are rendered by `02_analyse.py` into
# `data/downscaling/gifs/` - regenerable, not tracked in git; see
# `e2s/paths.py`'s `downscaling_gifs_path` docstring.
#
# The two gifs are paced to stay in simulated-time sync: the dense gif
# switches frames every 0.15s of simulated hour (matching the pacing of the
# main ensemble chapter's gifs), and the native-only gif's frames are
# proportionally longer (6x), since each one covers 6x as much simulated
# time. Both reach hour 24 at the same wall-clock moment, not just the
# same total length.

# %%
copy_asset(
    paths.downscaling_gifs_path / "member_00" / "t2m_robinson_dense.gif",
    paths.downscaling_book_path / "t2m_robinson_dense_member00.gif",
)
copy_asset(
    paths.downscaling_gifs_path / "member_00" / "t2m_robinson_native.gif",
    paths.downscaling_book_path / "t2m_robinson_native_member00.gif",
)

# %% [markdown]
# ```{figure} ../../output/downscaling/book/t2m_robinson_dense_member00.gif
# :name: fig-downscaling-t2m-gif-dense
# Member 0's 2m temperature field, hourly - each frame's title marks
# whether it's a native SFNO step or AFNO-interpolated.
# ```
#
# ```{figure} ../../output/downscaling/book/t2m_robinson_native_member00.gif
# :name: fig-downscaling-t2m-gif-native
# The same member, same rollout, only the native 6h SFNO frames - slower
# per frame, so it still covers the same 24h in the same wall-clock time
# as the animation above.
# ```
