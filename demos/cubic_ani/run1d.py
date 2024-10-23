###############################################################################
###
### Simple 1D demo to test the cubic anisotropy
###
###############################################################################

import numpy as np
from scipy import constants

import neuralmag as nm

nm.config.fem["n_gauss"] = 1

# setup state
mesh = nm.Mesh((50,), (2e-9, 1e-9, 1e-9))
state = nm.State(mesh)

state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.6e-11
state.material.Kc = -2.5e6
state.material.Kc_axis1 = [1, 0, 0]
state.material.Kc_axis2 = [0, 1, 0]
state.material.Kc_axis3 = [0, 0, 1]
state.material.Db = 3e-3
state.material.alpha = 0.1

# initial magnetization
wavelength = 1 * 4 * constants.pi * state.material.A / state.material.Db
x = state.coordinates("n")
state.m = nm.VectorFunction(
    state,
    tensor=state.tensor(
        np.stack(
            [
                0 * x[0],
                np.sin(2 * np.pi * x[0] / wavelength),
                np.cos(2 * np.pi * x[0] / wavelength),
            ],
            axis=-1,
        )
    ),
)

## register effective field
nm.ExchangeField().register(state, "exchange")
nm.CubicAnisotropyField().register(state, "aniso")
nm.BulkDMIField().register(state, "dmi")
nm.TotalField("exchange", "aniso", "dmi").register(state)

# relax
llg = nm.LLGSolver(state)
llg.step(10e-9)
state.write_vti("m", "m_out.vti")
