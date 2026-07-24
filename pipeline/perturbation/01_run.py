"""Compare ensemble-generation strategies for FCN3: perturbations applied
once to the initial condition (static noise, or dynamically "bred" via
short pre-rollout model integration) vs. a perturbation injected at
*every* step of the actual forecast rollout.

earth2studio.run.ensemble() calls its `perturbation` argument exactly
once, right before the rollout starts (see run.py: `x, coords =
perturbation(x, coords)`, called before `prognostic.create_iterator(...)`)
- none of the built-in Perturbation classes touch the state again once
the rollout is running. This holds even for the "Bred Vector" methods
(BredVector, HemisphericCentredBredVector): they *do* iterate the model
internally to grow a flow-consistent perturbation, but that breeding
happens before the rollout starts, seeding a single IC perturbation -
still one call from run.ensemble()'s perspective, just a more elaborate
one than IID noise.

True per-step perturbation therefore needs a hand-rolled rollout loop
(run_per_step below), calling the model's __call__ directly (one
autoregressive step, returns the next state) instead of
create_iterator() (an opaque generator that owns its internal state
end-to-end and can't be interrupted to modify x mid-rollout) - see
PrognosticModel.__call__ vs .create_iterator() in
earth2studio/models/px/base.py. run_per_step reimplements the relevant
slice of run.ensemble()'s internals (fetch_data + io.add_array + batched
rollout) to do this; checkpointing/resume, which the built-in workflow
supports, is intentionally left out - not needed for this comparison.

Reduced ensemble size/rollout length vs. pipeline/ensemble/01_run.py (4
members, 20 steps rather than 8/30): 7 configurations run back to back
here, so this keeps total GPU time roughly comparable to one run of the
main ensemble pipeline instead of ~7x it.
"""

import numpy as np
import torch

from earth2studio.data import GFS, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN3
from earth2studio.perturbation import (
    BredVector,
    Brown,
    Gaussian,
    HemisphericCentredBredVector,
    SphericalGaussian,
    Zero,
)
from earth2studio.run import ensemble as run_ensemble
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import to_time_array

from e2s.paths import ProjPaths

paths = ProjPaths()
paths.ensure_directories()

N_ENSEMBLE = 4
N_STEPS = 20
START_DATE = "2026-07-23T00:00:00"  # UTC - GFS timestamps are always UTC
N_BATCH_SIZE = 2

NOISE_AMPLITUDE = 0.05  # earth2studio's own default for Gaussian/Brown/BredVector
# Smaller than NOISE_AMPLITUDE: this one is injected fresh at every one of
# N_STEPS steps rather than just once, so it accumulates over the rollout -
# using the same amplitude as the IC-only methods would very likely blow
# the rollout up.
PER_STEP_NOISE_AMPLITUDE = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading FCN3 model weights...")
package = FCN3.load_default_package()
model = FCN3.load_model(package).to(device)

data = GFS()

# Perturbation methods usable directly with earth2studio's built-in
# run.ensemble() workflow - it applies each of these exactly once, to the
# initial condition, before the rollout starts. Covers both categories
# that only differ in *how* that single IC perturbation is constructed:
# static noise (zero/gaussian/brown/spherical_gaussian) and bred-vector
# methods (bred_vector/hemispheric_bred_vector), which use the model
# itself, iteratively, to grow a flow-consistent perturbation beforehand.
STANDARD_CONFIGS = {
    "zero": Zero(),
    "gaussian": Gaussian(noise_amplitude=NOISE_AMPLITUDE, seed=0),
    "brown": Brown(noise_amplitude=NOISE_AMPLITUDE),
    "spherical_gaussian": SphericalGaussian(noise_amplitude=NOISE_AMPLITUDE),
    "bred_vector": BredVector(
        model=model, noise_amplitude=NOISE_AMPLITUDE,
        seeding_perturbation_method=Brown(noise_amplitude=NOISE_AMPLITUDE),
    ),
    "hemispheric_bred_vector": HemisphericCentredBredVector(
        model=model, data=data,
        seeding_perturbation_method=Brown(noise_amplitude=NOISE_AMPLITUDE),
    ),
}


def run_standard(name, perturbation):
    zarr_path = paths.perturbation_zarr_path(name)
    print(f"\n=== {name}: IC-only perturbation via run.ensemble ===")
    io = ZarrBackend(str(zarr_path))
    run_ensemble(
        time=[START_DATE], nsteps=N_STEPS, nensemble=N_ENSEMBLE,
        prognostic=model, data=data, io=io, perturbation=perturbation,
        batch_size=N_BATCH_SIZE, device=device,
    )
    print(f"Saved '{name}' to {zarr_path}")


def run_per_step(name, step_perturbation):
    """Custom rollout: no IC perturbation at all by design (all members
    start from the identical, unperturbed analysis state), so any
    ensemble spread that emerges comes entirely from step_perturbation
    being re-applied after every one of the N_STEPS autoregressive
    steps - isolating what per-step injection alone contributes to
    spread growth, uncontaminated by a head start at t=0."""
    zarr_path = paths.perturbation_zarr_path(name)
    print(f"\n=== {name}: perturbation injected at every step ===")
    io = ZarrBackend(str(zarr_path))

    prognostic_ic = model.input_coords()
    time = to_time_array([START_DATE])
    x0, coords0 = fetch_data(
        source=data, time=time, variable=prognostic_ic["variable"],
        lead_time=prognostic_ic["lead_time"], device=device,
    )

    # Pre-allocate the zarr's full coordinate system - mirrors
    # earth2studio.run.ensemble()'s own setup (see run.py) so this store
    # has the same schema (ensemble/time/lead_time/variable/lat/lon) as
    # the ones written by run_standard() above.
    total_coords = model.output_coords(model.input_coords()).copy()
    if "batch" in total_coords:
        del total_coords["batch"]
    total_coords["time"] = time
    total_coords["lead_time"] = np.asarray(
        [model.output_coords(model.input_coords())["lead_time"] * i for i in range(N_STEPS + 1)]
    ).flatten()
    total_coords.move_to_end("lead_time", last=False)
    total_coords.move_to_end("time", last=False)
    total_coords = {"ensemble": np.arange(N_ENSEMBLE)} | total_coords
    variables_to_save = total_coords.pop("variable")
    io.add_array(total_coords, variables_to_save)

    batch_size = min(N_ENSEMBLE, N_BATCH_SIZE)
    for batch_id in range(0, N_ENSEMBLE, batch_size):
        mini_batch_size = min(batch_size, N_ENSEMBLE - batch_id)
        ensemble_coords = np.arange(batch_id, batch_id + mini_batch_size)
        print(f"Running batch starting at member {batch_id} ({mini_batch_size} members)...")

        x = x0.to(device)
        coords = {"ensemble": ensemble_coords} | coords0.copy()
        x = x.unsqueeze(0).repeat(mini_batch_size, *([1] * x.ndim))
        x, coords = map_coords(x, coords, prognostic_ic)

        io.write(*split_coords(x, coords))  # step 0: the unperturbed IC, identical across members

        for step in range(1, N_STEPS + 1):
            x, coords = model(x, coords)              # one autoregressive step
            x, coords = step_perturbation(x, coords)   # inject fresh noise before writing/feeding forward
            io.write(*split_coords(x, coords))

    print(f"Saved '{name}' to {zarr_path}")


for config_name, perturbation in STANDARD_CONFIGS.items():
    run_standard(config_name, perturbation)

run_per_step("per_step_brown", Brown(noise_amplitude=PER_STEP_NOISE_AMPLITUDE))

print("\nAll perturbation configurations complete.")
