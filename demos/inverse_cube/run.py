from nmagnum import *
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

mesh = Mesh((10, 10, 10), (5e-9, 5e-9, 5e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-11)
state.material.Ku = CellFunction(state).from_constant(1e5)
state.material.Ku_axis = VectorCellFunction(state).from_constant((0, 0, 1))
state.material.alpha = 0.1

state.m = VectorFunction(state).from_constant((0, 0, 1))
m0 = VectorFunction(state).from_constant((0, 0, 1)).tensor

# setup external field depending on phi and theta
state.phi = np.pi/2
state.theta = np.pi/2
h_ext = lambda phi, theta: torch.stack([
    2.4e5 * torch.sin(theta) * torch.cos(phi),
    2.4e5 * torch.sin(theta) * torch.sin(phi),
    2.4e5 * torch.cos(theta)
]).reshape((1,1,1,3)).expand((11,11,11,3))

# register effective field
ExchangeField().register(state, 'exchange')
UniaxialAnisotropyField().register(state, 'aniso')
ExternalField(h_ext).register(state, 'external')
TotalField('exchange', 'aniso', 'external').register(state)

# relax to s-state
llg = LLGSolver(state, parameters = ['phi', 'theta'])

# define optimization problem
m_target = VectorFunction(state).from_constant((np.sqrt(0.5), 0, np.sqrt(0.5)))

optimizer = torch.optim.Adam(llg.parameters(), lr = 0.05)

for epoch in range(10):
    print("epoch: ", epoch)
    optimizer.zero_grad()

    m_pred = odeint(llg, m0, state.tensor([0., 0.5e-9]))
    loss = torch.mean(torch.abs(m_pred[-1] - m_target))

    loss.backward()
    optimizer.step()

    print(state.phi)
    print(state.theta)
