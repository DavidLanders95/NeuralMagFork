###############################################################################
###
### MuMag Standard Problem #4 using a 2D mesh as introduced in
### https://www.ctcms.nist.gov/~rdm/std4/spec4.html
###
### This example is explained in details in the getting started section
### of the magnum.fe manual.
###
###############################################################################

import numpy as np

import neuralmag as nm

nm.config.backend = "jax"

# setup state
mesh = nm.Mesh((100, 25), (5e-9, 5e-9, 3e-9))
state = nm.State(mesh)

# setup material and m0
state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.alpha = 1.0

state.m = nm.VectorFunction(state).fill((np.sqrt(0.5), np.sqrt(0.5), 0))
h_ext = nm.VectorFunction(state).fill((-19576.0, 3421.0, 0.0), expand=True)

# register effective field
nm.ExchangeField().register(state, "exchange")
nm.DemagField().register(state, "demag")
nm.ExternalField(h_ext).register(state, "external")
nm.TotalField("exchange", "demag").register(state)

# relax to s-state
llg = nm.LLGSolver(state)
llg.relax()

# set external field and perform switch
state.material.alpha = 0.02
nm.TotalField("exchange", "demag", "external").register(state)
llg.reset()

logger = nm.Logger("data2d", ["t", "m", "E"], ["m"])
for i in range(100):
    logger.log(state)
    llg.step(1e-11)
