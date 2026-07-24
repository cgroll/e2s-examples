# E2S Examples

Experiments against [earth2studio](https://github.com/NVIDIA/earth2studio) AI
weather models, run as a [DVC](https://dvc.org/)-orchestrated pipeline and
published as a [MyST](https://mystmd.org/) book. Dependencies are managed by
[uv](https://docs.astral.sh/uv/).

## Experiments

| # | Experiment | Pipeline | Status |
|---|------------|----------|--------|
| 1 | Ensemble forecast + validation | `pipeline/ensemble/` | working |
| 2 | Temporal downscaling (6h → 1h via InterpModAFNO) | `pipeline/downscaling/` | starter script |
| 3 | Per-step variable subsetting (does it actually reduce compute?) | `pipeline/variable_subset/` | starter script |
| 4 | Diagnostic model (solar radiation from FCN3's state) | `pipeline/diagnostic/` | starter script |

See [todos.md](todos.md) for open questions and next steps per experiment.

## Setup

```bash
# Install uv if you haven't already — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

`uv sync` here only installs orchestration tooling (dvc, jupytext, mystmd) —
it deliberately does **not** install torch/earth2studio. See "Dependency
management" below for why, and what runs the actual pipeline stages.

## Dependency management

`earth2studio`'s own `pyproject.toml` pins several dependencies to specific
git revisions via `[tool.uv.sources]` (`makani`, `torch-harmonics`, a
`cfgrib` fork, …) because they aren't on PyPI or need an unreleased fix.
**That config only applies when earth2studio is the root of its own uv
project.** Consumed as a git dependency from a *different* project's
`pyproject.toml` (e.g. plain `uv add earth2studio[fcn3]`), uv ignores those
source overrides entirely and tries to resolve the same package names from
PyPI — which fails or pulls the wrong version for exactly the packages that
needed pinning in the first place.

[e2s-launchable](https://github.com/cgroll/e2s-launchable)'s Dockerfile
doesn't route around this with uv config either — it installs everything
imperatively and in a specific order (`torch` alone first, then
`torch-harmonics`/`makani` from git with `--no-build-isolation` so their
CUDA-extension build can see the already-installed `torch`, only then
`earth2studio` itself). That ordering isn't something a single `uv sync`
naturally reproduces even if the source pins are copied over.

So this project uses **two separate environments**:

| Environment | Contains | Managed by |
|-------------|----------|------------|
| `/root/earth2studio-project/.venv` | torch, earth2studio + extras, numpy/xarray/pandas/matplotlib/cartopy | the e2s-launchable Dockerfile (imperative `uv pip install`, not this repo) |
| this repo's own `.venv` | dvc, jupytext, mystmd, nbconvert, ipykernel | `uv sync` against `pyproject.toml` |

Every `dvc.yaml` stage invokes the GPU venv's interpreter directly
(`${E2S_PYTHON:-/root/earth2studio-project/.venv/bin/python}`, with
`PYTHONPATH=.` so `e2s/` is importable without installing it there) instead
of `uv run python ...`. `make run` (`uv run dvc repro`) still comes from
this repo's slim venv — only `dvc` itself needs to be on `uv`'s radar, not
what each stage's `cmd:` executes. Override the GPU interpreter with `export
E2S_PYTHON=/path/to/other/venv/bin/python` if that environment moves.

If you ever need `e2s-examples` to build standalone on a machine without
that pre-built venv (e.g. CI), the fallback is to copy the relevant
`[tool.uv.sources]` entries (`makani`, `torch-harmonics`, `cfgrib`) and the
`no-build-isolation-package` list for the extras actually used
(`fcn3`, `interp-modafno`, `precip-afno`, `solarradiation-afno`, `data`)
into this project's own `pyproject.toml`, and sync `torch` on its own before
syncing the rest — untested here, treat as a starting point, not a recipe.

## Running the pipeline

```bash
make dry-run   # preview what would run (dvc repro --dry)
make run       # execute everything that's out of date (dvc repro)
make serve     # http://localhost:3000 — live book preview
```

Or target a single experiment: `uv run dvc repro ensemble_run`.

`make serve` (`myst start`) runs two local servers, both bound to
`127.0.0.1`: the site itself on port 3000, and a second "content server" on
port 3100 that serves processed images — the page embeds absolute
`http://localhost:3100/...` URLs for them. If you're viewing the book
through SSH/VS Code port-forwarding and only forward 3000, the page loads
but every image is broken. Forward 3100 too:

```bash
ssh -L 3000:localhost:3000 -L 3100:localhost:3100 <remote>
```

The static build (`make build-book`, also what CI publishes to gh-pages)
doesn't have this problem — it has no second server, so it only needs one
forwarded port.

## Project layout

```
project-root/
├── e2s/                        # Python package — shared code
│   ├── paths.py                 # Centralized path config (ProjPaths)
│   └── validation.py            # Bounds/consistency-check helpers
├── pipeline/                   # Pipeline scripts, one subfolder per experiment
│   ├── ensemble/
│   ├── downscaling/
│   ├── variable_subset/
│   └── diagnostic/
├── book/                        # MyST book source
│   ├── notebooks/                # Executed notebooks (DVC output)
│   ├── markdown/                 # Static content
│   └── myst.yml                  # TOC and site settings
├── data/                         # Git-ignored, DVC-cached (zarr stores, gifs)
├── output/                       # Figures/tables/reports (tracked in git)
├── dvc.yaml                      # Pipeline DAG
└── dvc.lock                      # Pipeline state (checksums), tracked in git
```

See [contribution_conventions.md](contribution_conventions.md) for details on
adding pipeline stages and DVC/book conventions.

## Common DVC commands

| Command | Effect |
|---------|--------|
| `dvc repro --dry` | Dry run — show what would execute |
| `dvc repro` | Run pipeline (only rebuilds what's out of date) |
| `dvc repro -f <stage>` | Force-re-run a specific stage |
| `dvc repro <stage>` | Build one specific stage (and its dependencies) |
| `dvc dag` | Print the pipeline DAG |
