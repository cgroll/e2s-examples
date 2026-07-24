---
title: Welcome
---

# Earth2Studio Experiments

Four experiments run against [earth2studio](https://github.com/NVIDIA/earth2studio) models:

1. **Ensemble forecast + validation** — FCN3 ensemble rollout with automated
   standalone/cross-variable/cross-time/cross-ensemble consistency checks.
   See `pipeline/ensemble/`.
2. **Temporal downscaling** — interpolate a 6h-native model to 1h steps via
   `InterpModAFNO`. See `pipeline/downscaling/`.
3. **Per-step variable subsetting** — verify whether writing only a subset of
   variables at each rollout step actually reduces compute, not just storage.
   See `pipeline/variable_subset/`.
4. **Diagnostic model** — derive a variable that isn't part of the
   prognostic model's state (precipitation or solar radiation) via
   `earth2studio.run.diagnostic`. See `pipeline/diagnostic/`.

## How to read this book

Each chapter corresponds to a `pipeline/<experiment>/` directory. Chapters
are added to the table of contents in `book/myst.yml` once an experiment has
an executed notebook to show — see `contribution_conventions.md` for the
convention pipeline scripts follow to become book chapters.
