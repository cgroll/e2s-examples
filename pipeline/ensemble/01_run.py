import numpy as np
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
START_DATE = '2026-07-23T00:00:00'
N_BATCH_SIZE = 2

# Variables to keep in the zarr output. FCN3 still computes its full 72-variable
# state internally at every step (it needs the whole state to roll forward), so
# this only cuts storage/write time, not compute. Downstream (02_validate.py,
# 03_validate_visualize.py, 04_analyse.py) only checks/plots whatever variables
# are actually present, so trimming this list silently disables the checks that
# depend on what's missing - see the printed [WARN] lines in 02_validate.py.
OUTPUT_VARIABLES = ["u10m", "v10m", "t2m", "z500"]

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
    output_coords={"variable": np.array(OUTPUT_VARIABLES)},
)
print(f"Ensemble forecast complete! Saved to {zarr_path}")