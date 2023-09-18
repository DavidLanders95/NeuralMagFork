from nmagnum import *
import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
state = State(mesh)

state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-11)
state.material.alpha = 1.0

state.m = VectorFunction(state).from_constant((1./np.sqrt(2.), 1./np.sqrt(2.), 0))
state.h_ext = VectorFunction(state).from_constant((-19576., 3421., 0.))

# TOP
ExchangeField().register(state)
DemagField().register(state)
TotalField('h_exchange', 'h_demag').register(state)

llg = LLGSolver(state)

llg.step(1e-9)

print(state.m.avg())
state.m.write("m0.vti")

TotalField('h_exchange', 'h_demag', 'h_ext').register(state)
state.material.alpha = 0.02
llg = LLGSolver(state)

with open('log.dat', 'w') as f:
    for i in range(100):
        llg.step(1e-11)
        avg = state.m.avg()
        f.write(f"{state.t} {avg[0]} {avg[1]} {avg[2]}\n")
