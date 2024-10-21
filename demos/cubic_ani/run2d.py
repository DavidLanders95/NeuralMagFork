###############################################################################
###
### Testing cubic anisotropy
###
###############################################################################

from scipy import constants

import neuralmag as nm

nm.config.fem["n_gauss"] = 1

# setup state
mesh = nm.Mesh((50, 50), (2e-9, 2e-9, 0.6e-9), (-50e-9, -50e-9, 0))
state = nm.State(mesh)

state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.6e-11
state.material.Kc = 510e3
state.material.Kc_axis1 = [1, 0, 0]
state.material.Kc_axis2 = [0, 1, 0]
state.material.Kc_axis3 = [0, 0, 1]
state.material.alpha = 0.1

# initial magnetization
state.m = nm.VectorFunction(state).fill((0, 0.1, 1))

## register effective field
nm.ExchangeField().register(state, "exchange")
nm.CubicAnisotropyField().register(state, "aniso")
nm.TotalField("exchange", "aniso").register(state)

# relax to skyrmion
llg = nm.LLGSolver(state)
llg.step(1e-9)
state.write_vti("m", "m_out.vti")
