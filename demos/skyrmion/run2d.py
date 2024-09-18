###############################################################################
###
### Stabilization of a DMI stabilized Skyrmion in a thin disk
###
###############################################################################

from scipy import constants

import neuralmag as nm

# setup state
mesh = nm.Mesh((50, 50), (2e-9, 2e-9, 0.6e-9), (-50e-9, -50e-9, 0))
state = nm.State(mesh)

state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.6e-11
state.material.Di = 4e-3
state.material.Di_axis = [0, 0, 1]
state.material.Ku = 510e3
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 0.1

# set circular geometry
state.rho = nm.CellFunction(state)
x, y = state.coordinates()
state.rho.tensor = nm.config.backend.np.where(
    x**2.0 + y**2.0 < 50e-9**2.0, 1.0, state.eps
)

# initial magnetization
state.m = nm.VectorFunction(state).fill((0, 0, 1))

## register effective field
nm.ExchangeField().register(state, "exchange")
nm.DemagField().register(state, "demag")
nm.InterfaceDMIField().register(state, "dmi")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.TotalField("exchange", "demag", "dmi", "aniso").register(state)

# relax to skyrmion
llg = nm.LLGSolver(state)
llg.step(1e-9)
state.write_vti(["m", "rho"], "skyrmion.vti")
