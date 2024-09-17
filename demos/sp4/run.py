###############################################################################
###
### MuMag Standard Problem #4 as introduced in
### https://www.ctcms.nist.gov/~rdm/std4/spec4.html
###
### This example is explained in details in the getting started section
### of the magnum.fe manual.
###
###############################################################################

import neuralmag as nm

nm.config.backend = "jax"

# setup mesh and state
mesh = nm.Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
state = nm.State(mesh)

# setup material and m0
state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.alpha = 1.0

# initialize nodal vector functions for magneization and external field
state.m = nm.VectorFunction(state).fill((0.5**0.5, 0.5**0.5, 0))
h_ext = nm.VectorFunction(state).fill([-19576.0, 3421.0, 0.0], expand=True)

# register effective field contributions
nm.ExchangeField().register(state, "exchange")
nm.DemagField().register(state, "demag")
nm.ExternalField(h_ext).register(state, "external")
nm.TotalField("exchange", "demag").register(state)

# relax to s-state
llg = nm.LLGSolver(state)
llg.relax()

state.write_vti("m", "sstate.vti")

# set external field and damping to perform switch
nm.TotalField("exchange", "demag", "external").register(state)
state.material.alpha = 0.02
state.t = 0.0
llg.reset()

logger = nm.Logger("data", ["t", "m"], ["m"])
while state.t < 1e-9:
    logger.log(state)
    llg.step(1e-11)
