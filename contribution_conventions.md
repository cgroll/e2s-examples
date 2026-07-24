# Contribution Conventions

This document describes the project structure and conventions. It is
written for human contributors and AI agents alike.

## Project Structure

```
project-root/
├── e2s/                        # Python package — shared utilities
│   ├── __init__.py
│   ├── paths.py                 # Centralized path configuration (ProjPaths)
│   └── validation.py             # Bounds/consistency-check helpers, shared
│                                  # across experiments' validation stages
├── pipeline/                    # Pipeline scripts, one subfolder per experiment
│   ├── ensemble/                 # 01_run → 02_validate → 03_validate_visualize → 04_analyse
│   ├── downscaling/
│   ├── variable_subset/
│   └── diagnostic/
├── book/                         # MyST Jupyter Book source
│   ├── notebooks/                 # Executed .ipynb files (produced by DVC)
│   ├── markdown/                  # Static hand-written content
│   └── myst.yml                   # Book configuration and table of contents
├── data/                          # Git-ignored, DVC-cached, one dir per experiment
│   ├── <experiment>/*.zarr        # Prognostic/diagnostic model outputs
│   └── ensemble/gifs/             # Regenerable per-member animations (too
│                                    # large for git even as a tracked figure)
├── output/                        # Tracked in git, one dir per experiment
│   ├── <experiment>/               # Small figures/tables meant to be read,
│                                    # not regenerated to view
├── dvc.yaml                       # Pipeline definition (stages, deps, outs)
├── dvc.lock                       # Pipeline state (checksums) — tracked in git
└── pyproject.toml                 # Orchestration deps only — see "Two environments"
```

## Tools

| Tool | Purpose |
|------|---------|
| **uv** | Package and environment management — **for this repo's own orchestration deps only** (dvc, jupytext, mystmd, …), not torch/earth2studio. See "Two environments" below. |
| **DVC** | Pipeline orchestration (dependency-aware task runner) |
| **jupytext** | Execute `.py` analysis scripts → `.ipynb` notebooks (for book chapters) |
| **MyST / mystmd** | Build the HTML book from notebooks and markdown |

## Two environments

`uv add earth2studio` from a consuming project doesn't work: earth2studio
pins several dependencies (`makani`, `torch-harmonics`, a `cfgrib` fork) to
git revisions via `[tool.uv.sources]` in its own `pyproject.toml`, and that
config only applies when earth2studio is the root of its own uv project —
not when it's someone else's git dependency. See "Dependency management" in
`README.md` for the full explanation.

Consequence for every pipeline script: it runs against
`/root/earth2studio-project/.venv` (built by the
[e2s-launchable](https://github.com/cgroll/e2s-launchable) Dockerfile, not
by anything in this repo), invoked directly in `dvc.yaml` via
`${E2S_PYTHON:-/root/earth2studio-project/.venv/bin/python}` with
`PYTHONPATH=.` — never via `uv run python pipeline/...`. `uv run` is only
correct for tools this repo's own `pyproject.toml` actually declares (`dvc`,
`jupytext`, `mystmd`). When adding a new stage, copy the `PYTHONPATH=.
${E2S_PYTHON:-...}` prefix from an existing stage rather than reaching for
`uv run python`.

## DVC Primer

[DVC](https://dvc.org/) reads `dvc.yaml` in the project root and tracks
stage state in `dvc.lock`. Each stage declares *how* to produce outputs from
dependencies; DVC hashes `cmd` and every `deps` entry and compares against
`dvc.lock` — if nothing changed, the stage is skipped.

### `cache: false` for git-tracked outputs

By default DVC moves stage outputs into its own cache and git-ignores
them — appropriate for `data/`. But small figures/tables under `output/`
should stay as normal files tracked directly by git, so every such output is
declared with `cache: false`. DVC still hashes the file to detect
staleness; it just doesn't duplicate it into `.dvc/cache`.

### `persist: true` for model rollouts

Zarr stores from a prognostic/diagnostic run are expensive to regenerate and
should not be silently wiped and re-run. Mark them `persist: true` so DVC
leaves the existing file in place whenever the stage is (re-)run. To force a
fresh run: delete the store, or run `dvc repro -f <stage>`.

### Data vs. output: the size test

Decide `data/` vs. `output/` by asking "would I want this file to show up in
a `git diff` on every re-run?":
- **Large or binary** (zarr stores, per-member gif animations) → `data/`,
  DVC-cached, git-ignored.
- **Small and meant to be read** (PNG figures, CSV tables, markdown
  reports, a few hundred KB at most) → `output/`, `cache: false`, tracked in
  git so the book/PR can be reviewed without re-running the pipeline.

This is why `output/ensemble/analysis/` holds only the Munich meteogram
PNGs, while the much larger per-member Robinson-projection gifs live in
`data/ensemble/gifs/` — see `e2s/paths.py`'s `ensemble_gifs_path` docstring.

### Running the pipeline

```bash
dvc repro --dry        # dry-run: show what would execute
dvc repro               # run the full pipeline (skips up-to-date stages)
dvc repro <stage>       # build one specific stage (and its dependencies)
dvc repro -f <stage>    # force-re-run a specific stage
dvc dag                 # print the pipeline DAG
```

Or via Make shortcuts: `make run`, `make dry-run`, `make serve`.

## Pipeline Conventions

### Numbering within an experiment folder

- `01_run*` — the expensive GPU inference step (prognostic/diagnostic model
  run), writes to `data/<experiment>/`.
- `02_validate*` — pure data checks against the run's output, writes tables
  to `output/<experiment>/`.
- `03_validate_visualize*` / `04_analyse*` — figures, either QA plots from
  the validation tables or narrative analysis plots.

Not every experiment needs all four; `variable_subset` for example is a
single comparison script, not a rollout + validate + visualize chain.

### Path conventions

All scripts must be runnable from any working directory. Use `ProjPaths`
from `e2s/paths.py`:

```python
from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

io = ZarrBackend(str(paths.ensemble_zarr_path))
fig.savefig(paths.ensemble_analysis_path / "meteogram.png")
```

Add a `@property` to `e2s/paths.py` for every new data/output file a stage
introduces, following the existing per-experiment grouping.

### Book integration (not yet wired up)

None of the pipeline scripts currently produce book chapters — they're
plain `.py` scripts, not jupytext-formatted analysis notebooks. To turn one
into a chapter once an experiment has something worth narrating:

1. Add a jupytext header (`# %%` cell markers, kernelspec) — see the
   [project-book-template-dvc](https://github.com/cgroll/project-book-template-dvc)
   for the exact format and a worked example.
2. Add a DVC stage that runs `jupytext --to notebook --execute` into
   `book/notebooks/`. Because the script being executed imports
   torch/earth2studio, `--execute` needs a Jupyter kernel backed by
   `/root/earth2studio-project/.venv`, not this repo's own venv — register
   one once with `${E2S_PYTHON:-/root/earth2studio-project/.venv/bin/python}
   -m ipykernel install --user --name=e2s-gpu` and pass `--set-kernel
   e2s-gpu`. `jupytext` itself can still come from `uv run` (this repo's
   slim venv); only the kernel that executes the notebook cells needs to be
   the GPU one.
3. Add the notebook to `book/myst.yml`'s `toc`.

## Adding a New Pipeline Stage

1. **Write the script** in `pipeline/<experiment>/`.
2. **Add a property** to `e2s/paths.py` for every new data/output file.
3. **Add a stage** to `dvc.yaml` with `cmd`, `deps`, and `outs` (`cache:
   false` for anything under `output/`, `persist: true` for model
   rollouts under `data/`). Keep stage-owned output directories
   non-overlapping — DVC rejects a path that's nested inside another
   stage's declared output. `cmd` should start with `PYTHONPATH=.
   ${E2S_PYTHON:-/root/earth2studio-project/.venv/bin/python}` (copy from an
   existing stage) — see "Two environments" above.
4. **(Optional)** turn it into a book chapter — see above.

## Git Conventions

| Tracked | Not tracked |
|---------|-------------|
| `pipeline/**/*.py` source files | `data/**` |
| `output/**` figures, tables, reports | `.venv/` |
| `book/notebooks/*.ipynb`, `book/markdown/*.md` | `book/_build/` |
| `dvc.yaml`, `dvc.lock` | |
| `e2s/**/*.py` | |
