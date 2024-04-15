import torch
from scipy import constants

from neuralmag import *

# setup state
mesh = Mesh((50, 50), (2e-9, 2e-9, 0.6e-9), (-50e-9, -50e-9, 0))
state = State(mesh)

state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.6e-11
state.material.Di = 4e-3
state.material.Di_axis = [0, 0, 1]
state.material.Ku = 510e3
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 0.1

# set circular geometry
state.rho = CellFunction(state).fill(state.eps)
x, y = state.coordinates()
state.rho.tensor[x**2.0 + y**2.0 < 50e-9**2.0] = 1.0

# initial magnetization
state.m = VectorFunction(state).fill((0, 0, 1))

## register effective field
ExchangeField().register(state, "exchange")
DemagField().register(state, "demag")
InterfaceDMIField().register(state, "dmi")
UniaxialAnisotropyField().register(state, "aniso")
TotalField("exchange", "demag", "dmi", "aniso").register(state)

# relax to skyrmion
llg = LLGSolver(state)
llg.step(1e-9)
state.write_vti(["m", "rho"], "skyrmion.vti")
