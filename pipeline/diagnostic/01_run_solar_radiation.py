"""Diagnostic model: derive surface solar radiation from SFNO's prognostic
state via SolarRadiationAFNO6H — a variable no earth2studio prognostic
model forecasts directly.

Uses SFNO, not FCN3: SolarRadiationAFNO6H's required 24 input variables
include `sp` (surface pressure), which FCN3 doesn't output (only `msl`) -
same incompatibility as pipeline/downscaling/01_run.py, see that script's
docstring for the full explanation. SFNO's output is a superset of what
SolarRadiationAFNO6H needs.

Useful downstream for PV-generation-potential estimates. For precipitation
instead, swap in earth2studio.models.dx.precipitation_afno.PrecipitationAFNO
(same load_default_package()/load_model() pattern) - unverified whether it
has the same sp-related constraint against FCN3.

earth2studio.run.diagnostic couples a single deterministic prognostic
rollout with a diagnostic model applied at every step - no ensemble/
perturbation here, this is about the diagnostic coupling, not spread.
"""

from earth2studio.models.px import SFNO
from earth2studio.models.dx import SolarRadiationAFNO6H
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import diagnostic as run_diagnostic

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

# --- Compat shim: see pipeline/downscaling/01_run.py for the full
# explanation - makani (pinned commit d473b054) requires
# params.img_shape_x_resampled/img_shape_y_resampled, which earth2studio's
# SFNO wrapper never sets. Only needed because we load SFNO here.
import makani.models.model_registry as _makani_model_registry

_original_get_model = _makani_model_registry.get_model


def _get_model_compat(params, *args, **kwargs):
    if not hasattr(params, "img_shape_x_resampled"):
        params.img_shape_x_resampled = params.img_shape_x
    if not hasattr(params, "img_shape_y_resampled"):
        params.img_shape_y_resampled = params.img_shape_y
    return _original_get_model(params, *args, **kwargs)


_makani_model_registry.get_model = _get_model_compat

N_STEPS = 12  # 6h-native steps -> 12 = 3 days
START_DATE = "2026-07-23T00:00:00"  # UTC - GFS timestamps are always UTC

# 1. Load the prognostic model
print("Loading SFNO model weights...")
px_package = SFNO.load_default_package()
prognostic = SFNO.load_model(px_package)

# 2. Load the diagnostic model - derives solar radiation from the
# prognostic state, not from anything the prognostic model outputs itself.
print("Loading SolarRadiationAFNO6H weights...")
dx_package = SolarRadiationAFNO6H.load_default_package()
diagnostic = SolarRadiationAFNO6H.load_model(dx_package)

# 3. Select initial condition data source
data = GFS()

# 4. Configure output backend
zarr_path = paths.diagnostic_zarr_path
io = ZarrBackend(str(zarr_path))

# 5. Execute diagnostic workflow
print("Running SFNO + SolarRadiationAFNO6H diagnostic workflow on GPU...")
run_diagnostic(
    time=[START_DATE],
    nsteps=N_STEPS,
    prognostic=prognostic,
    diagnostic=diagnostic,
    data=data,
    io=io,
)
print(f"Diagnostic run complete! Saved to {zarr_path}")
