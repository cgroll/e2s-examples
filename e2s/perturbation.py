"""Perturbation helpers shared between pipeline/perturbation/01_run.py's
IC-perturbation sweep and pipeline/downscaling/01_run.py's ensemble.

`Brown`/`Gaussian` (from earth2studio.perturbation) take a single
`noise_amplitude` applied identically to every variable in the state.
That's fine when every variable shares roughly the same physical scale,
but SFNO's 73-variable state spans several orders of magnitude (z500 ~
55000 m^2/s^2, u10m ~1-10 m/s) - a flat amplitude is simultaneously
negligible for the large-magnitude variables and catastrophic for the
small ones. `pipeline/perturbation/01_run.py`'s sweep diagnosed this
directly: a flat `noise_amplitude=0.05` applied to all variables blew up
SFNO within one step, while the same nominal 0.05 calibrated per
variable (this module) produced a clean, physically valid ensemble at
every intensity tested (see that sweep's `brown_0.05`/`brown_0.01`/
`brown_0.002` chapters). `compute_variable_scales`/
`ScaledBrownPerturbation` are the reusable pieces of that fix.
"""

import torch
from earth2studio.perturbation import Brown


def compute_variable_scales(x0, coords0):
    """Per-variable characteristic scale (spatial standard deviation,
    computed directly from x0, an unperturbed initial-condition tensor)
    used to calibrate ScaledBrownPerturbation's amplitude proportionally
    to each variable's own physical variability, instead of one flat
    number applied identically to all variables regardless of scale.

    coords0 must be the CoordSystem dict matching x0 exactly (same key
    order as x0's dims) - used only to find which dim is "variable"."""
    var_names = list(coords0["variable"])
    var_axis = list(coords0.keys()).index("variable")
    reduce_dims = tuple(d for d in range(x0.dim()) if d != var_axis)
    std = x0.std(dim=reduce_dims)
    return {name: max(std[i].item(), 1e-6) for i, name in enumerate(var_names)}


class ScaledBrownPerturbation:
    """Brown (spatially-correlated, reddened) noise whose amplitude is
    `intensity * variable_scales[variable]` per variable, instead of one
    flat noise_amplitude shared across all variables - see
    compute_variable_scales for why the flat version doesn't work.
    `intensity` is a dimensionless fraction of each variable's own
    spatial std, so the same intensity value (e.g. 0.05) means a
    comparable *relative* perturbation strength for every variable, not a
    comparable *absolute* one."""

    def __init__(self, variable_scales, intensity, reddening=2):
        self.variable_scales = variable_scales
        self.intensity = intensity
        self._noise_source = Brown(noise_amplitude=1.0, reddening=reddening)

    def __call__(self, x, coords):
        var_names = list(coords["variable"])
        var_axis = list(coords.keys()).index("variable")
        amp_shape = [1] * x.dim()
        amp_shape[var_axis] = len(var_names)
        amplitude = torch.tensor(
            [self.intensity * self.variable_scales[v] for v in var_names],
            device=x.device, dtype=x.dtype,
        ).reshape(amp_shape)
        noise = self._noise_source._generate_noise_correlated(tuple(x.shape), device=x.device)
        return x + amplitude * noise, coords


class SingleVariablePerturbation:
    """Wraps a base perturbation (e.g. Brown, ScaledBrownPerturbation) and
    applies it to only one named variable's channel, leaving every other
    variable exactly at its raw IC value."""

    def __init__(self, variable_name, base_perturbation):
        self.variable_name = variable_name
        self.base_perturbation = base_perturbation

    def __call__(self, x, coords):
        x_perturbed, coords = self.base_perturbation(x, coords)
        var_names = list(coords["variable"])
        var_axis = list(coords.keys()).index("variable")
        idx = var_names.index(self.variable_name)
        out = x.clone()
        out.select(var_axis, idx).copy_(x_perturbed.select(var_axis, idx))
        return out, coords
