import numpy as np
from scipy import constants

import neuralmag as nm

# nm.config.torch['compile'] = False
nm.config.fem["n_gauss"] = 1

# setup state
mesh = nm.Mesh((100,), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)
# state.spin_one = lambda m: m[0,:,2].sum()/2.

# setup material and m0
state.material.Ms = 0.86e6
state.material.A = 1.3e-11
state.material.Ku = 0.4e6
state.material.Ku_axis = [0, 0, 1]
state.material.Db = 3e-3
state.material.Di = 3e-3
state.material.Di_axis = [0, 0, 1]
state.material.alpha = 1.0

# print(state.material.A.shape)
# print(state.rho.shape)
# print(state.material.A[:])

state.m = nm.VectorFunction(state).fill((0, 0, 1))

# register effective field
# nm.BulkDMIField().register(state, "dmi")
nm.InterfaceDMIField().register(state, "dmi")
nm.ExchangeField().register(state, "exchange")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.TotalField("aniso", "dmi", "exchange").register(state)

# relax
llg = nm.LLGSolver(state)
llg.step(2e-9)

data = np.zeros((state.m.tensor.shape[0], 4))
data[:, 0] = np.arange(data.shape[0]) * 1e-9
data[:, (1, 2, 3)] = state.m.tensor[:, :]
np.savetxt("log.dat", data)

# state.write_vti(["m"], "m.vti")
