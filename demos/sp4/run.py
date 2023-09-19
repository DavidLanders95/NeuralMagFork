from nmagnum import *
import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

# setup state
mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-11)
state.material.alpha = 1.
state.h_ext = VectorFunction(state).from_constant((0, 0, 0))
state.m = VectorFunction(state).from_constant((1./np.sqrt(2.), 1./np.sqrt(2.), 0))

# register effective field
ExchangeField().register(state, 'h_exchange')
DemagField().register(state, 'h_demag')
TotalField('h_exchange', 'h_demag', 'h_ext').register(state, 'h')

# relax to s-state
llg = LLGSolver(state)
llg.step(1e-9)

# set external field and perform switch
state.h_ext.tensor[...,:] = state.tensor([-19576., 3421., 0.])
state.material.alpha = 0.02

with open('log.dat', 'w') as f:
    for i in range(100):
        llg.step(1e-11)
        avg = state.m.avg()
        f.write(f"{state.t} {avg[0]} {avg[1]} {avg[2]}\n")
