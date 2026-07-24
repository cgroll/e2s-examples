import torch
import xarray as xr
import matplotlib.pyplot as plt

from earth2studio.models.px import FCN3
from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.run import ensemble as run_ensemble
from earth2studio.perturbation import Zero

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

MODEL = 'FCN3'
N_ENSEMBLE = 8
N_STEPS = 30
START_DATE = '2026-07-23T00:00:00'  # UTC - GFS timestamps are always UTC
N_BATCH_SIZE = 2

# 1. Load pre-trained FCN3 model
print("Loading FCN3 model weights...")
package = FCN3.load_default_package()
model = FCN3.load_model(package)

# 2. Select initial condition data source
data = GFS()

# 3. Configure output backend
zarr_path = paths.ensemble_zarr_path
io = ZarrBackend(str(zarr_path))

# 4. Define the Perturbation Method
perturbation = Zero()

# 5. Execute Ensemble Forecast
#
# No output_coords override: writes FCN3's full 72-variable state (all
# pressure levels) rather than a subset, so downstream cross-variable
# consistency checks (e.g. z500 > z700 > z850 monotonicity in 02_validate.py)
# have more than one pressure level to compare.
print("Running FCN3 ensemble on GPU...")
run_ensemble(
    time=[START_DATE],
    nsteps=N_STEPS,
    nensemble=N_ENSEMBLE,
    prognostic=model,
    data=data,
    io=io,
    perturbation=perturbation,
    batch_size=N_BATCH_SIZE,
)
print(f"Ensemble forecast complete! Saved to {zarr_path}")