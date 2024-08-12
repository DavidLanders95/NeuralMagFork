###############################################################################
###
### Test case for the antiferromagnetic interface coupling. A soft magnetic
### layer is antiferromagnetically coupled to a hard layer and the equilirbium
### magnetization of the soft layer is computed for different external
### field strengths.
###
###############################################################################

from scipy import constants

import neuralmag as nm

# setup mesh and state
mesh = nm.Mesh((10, 10, 3), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)

# setup material and m0
state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.3e-11
state.material.alpha = 1.0
state.material.iA = nm.Function(state, "ccn").fill(-0.5e-3, expand=True)
state.material.Ku = nm.CellFunction(state).fill(1e7)
state.material.Ku.tensor[:, :, :2] = 0
state.material.Ku_axis = [1, 0, 0]

# set empty spacer layer
state.rho = nm.CellFunction(state).fill(1.0)
state.rho.tensor[:, :, 1] = state.eps

# initialize nodal vector functions for magnetization and external field
state.m = nm.VectorFunction(state).fill((1, 0, 0))
state.m.tensor[:, :, 2:, 0] = -1
state.write_vti(["m", "rho"], "m0.vti")

# register effective field contributions
nm.ExchangeField().register(state, "exchange")
nm.InterlayerExchangeField(1, 2).register(state, "rkky")
nm.ExternalField(
    lambda t: t * state.tensor([0, 5.0 / constants.mu_0 / 5e-9, 0])
).register(state, "external")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.TotalField("exchange", "rkky", "external", "aniso").register(state)

# register dynamic attributes for logging of magnetization in top and bottom layer
state.ma = lambda m: m[:, :, :2, :].mean(dim=(0, 1, 2))
state.mb = lambda m: m[:, :, 2:, :].mean(dim=(0, 1, 2))

# perform time intetgration
llg = nm.LLGSolver(state)
logger = nm.ScalarLogger("log.dat", ["h_external", "ma", "mb"])
while state.t < 5e-9:
    logger.log(state)
    llg.step(1e-11)
