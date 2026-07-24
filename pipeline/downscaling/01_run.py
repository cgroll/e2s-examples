"""Temporal downscaling: interpolate SFNO's native 6h steps to 1h steps.

InterpModAFNO wraps a base prognostic model (px_model) and inserts
`num_interp_steps` interpolated 1h steps between each of its native 6h
steps. Wrapped this way, it behaves as a normal PrognosticModel with a 1h
lead-time spacing, so it plugs into run.ensemble/run.deterministic exactly
like FCN3 does in pipeline/ensemble/01_run.py.

Uses SFNO, not FCN3: InterpModAFNO's expected 73-variable schema (checked
against earth2studio/models/px/interpmodafno.py's VARIABLES) matches SFNO's
output exactly, including `sp` (surface pressure). FCN3 only has 72 of
those - it has `msl` but not `sp` - and fails at the first inference step
with "Some elements of [...] are not in [...]" if paired with InterpModAFNO.
So this experiment downscales SFNO, a sibling model to FCN3 in earth2studio,
rather than the ensemble baseline model itself.

Reference: https://arxiv.org/abs/2410.18904
"""

import numpy as np

from earth2studio.models.px import SFNO, InterpModAFNO
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import ensemble as run_ensemble
from earth2studio.perturbation import Zero

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

# --- Compat shim: makani (pinned commit d473b054, installed for FCN3) added
# a hard requirement in model_registry.get_model() for
# params.img_shape_x_resampled/img_shape_y_resampled. earth2studio's SFNO
# wrapper - both the installed version and current main branch as of this
# writing - never sets these; it only sets img_local_shape_x/y from
# img_shape_x/y two lines above where get_model is called. The SFNO
# checkpoint downloaded from NGC only has img_shape_x/y in its config, so
# SFNO.load_model() crashes with AttributeError before this patch. Mirrors
# earth2studio's own img_local_shape_x/y pattern; lives only in this script,
# does not touch the shared venv on disk.
import makani.models.model_registry as _makani_model_registry

_original_get_model = _makani_model_registry.get_model


def _get_model_compat(params, *args, **kwargs):
    if not hasattr(params, "img_shape_x_resampled"):
        params.img_shape_x_resampled = params.img_shape_x
    if not hasattr(params, "img_shape_y_resampled"):
        params.img_shape_y_resampled = params.img_shape_y
    return _original_get_model(params, *args, **kwargs)


_makani_model_registry.get_model = _get_model_compat

N_ENSEMBLE = 8
N_HOURLY_STEPS = 24  # 1h steps -> 24 = one day of hourly forecast
START_DATE = "2026-07-23T00:00:00"
# InterpModAFNO's interpolation step appears to not broadcast its zenith-time
# modulation embedding across a batch >1 (RuntimeError: scale tensor has
# half the expected elements at batch_size=2) - run members one at a time
# until that's confirmed/fixed upstream.
N_BATCH_SIZE = 1

OUTPUT_VARIABLES = ["u10m", "v10m", "t2m", "z500"]

# 1. Load the base 6h prognostic model, then wrap it for 1h interpolation.
print("Loading SFNO base model weights...")
sfno_package = SFNO.load_default_package()
sfno = SFNO.load_model(sfno_package)

print("Loading InterpModAFNO interpolation weights...")
interp_package = InterpModAFNO.load_default_package()
model = InterpModAFNO.load_model(interp_package, px_model=sfno)

# 2. Select initial condition data source
data = GFS()

# 3. Configure output backend
zarr_path = paths.downscaling_zarr_path
io = ZarrBackend(str(zarr_path))

# 4. Define the Perturbation Method
perturbation = Zero()

# 5. Execute ensemble forecast at 1h resolution
print("Running SFNO + InterpModAFNO ensemble on GPU...")
run_ensemble(
    time=[START_DATE],
    nsteps=N_HOURLY_STEPS,
    nensemble=N_ENSEMBLE,
    prognostic=model,
    data=data,
    io=io,
    perturbation=perturbation,
    batch_size=N_BATCH_SIZE,
    output_coords={"variable": np.array(OUTPUT_VARIABLES)},
)
print(f"Downscaled ensemble forecast complete! Saved to {zarr_path}")

# TODO: reuse e2s.validation + pipeline/ensemble/02_validate.py's checks
# against this zarr store (STEP_JUMP_LIMITS etc. need retuning for 1h steps
# — they were calibrated for 6h transitions) to confirm the interpolated
# steps are physically consistent, not just interpolated-smooth.
