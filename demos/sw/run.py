from scipy import constants

import neuralmag as nm

nm.config.fem["n_gauss"] = 1

mesh = nm.Mesh((2, 2, 2), (5e-9, 5e-9, 5e-9))
state = nm.State(mesh)

# setup material and m0
state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.Ku = 1e5
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 1.0

state.m = nm.VectorFunction(state).fill((0, 0, 1))

# setup external field depending t
Hc = 2 * 1e5 / (constants.mu_0 * 8e5)

# register effective field
nm.ExchangeField().register(state, "exchange")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.ExternalField(lambda t: t * state.tensor([0, 1.2 / 10e-9 * Hc, 0])).register(
    state, "external"
)
nm.TotalField("exchange", "aniso", "external").register(state)

# relax
llg = nm.LLGSolver(state)

# foward
logger = nm.Logger("sw", ["t", "h_external", "m", "E"], ["m"])
while state.t < 10e-9:
    logger.log(state)
    llg.step(1e-11)
