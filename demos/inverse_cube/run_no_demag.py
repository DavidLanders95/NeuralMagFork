from nmagnum import *
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

config.fem['n_gauss'] = 1

mesh = Mesh((10, 10, 10), (5e-9, 5e-9, 5e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-11)
state.material.Ku = CellFunction(state).from_constant(1e5)
state.material.Ku_axis = VectorCellFunction(state).from_constant((0, 0, 1))
state.material.alpha = 0.1

state.m = VectorFunction(state).from_constant((0, 0, 1))

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

# relax
llg = LLGSolver(state, parameters = ['phi', 'theta'])

#with torch.no_grad():
#    llg.step(1e-9)

## CHECK RESULT
#state.phi = 1.475641455028714466e+00
#state.theta = 1.497402584324188712e+00
#
#llg.step(0.5e-9)
#print(state.m.avg())
#state.write_vti(state.m, 'm0.vtu')
#exit()
## END CHECK RESULT

# define optimization problem
m_target = VectorFunction(state).from_constant((np.sqrt(0.5), 0, np.sqrt(0.5))).tensor

optimizer = torch.optim.Adam(llg.parameters(), lr = 0.05)

with open('log.dat', 'w') as f:
    for epoch in range(100):
        print(f"epoch: {epoch}")
        optimizer.zero_grad()

        m_pred = odeint(llg, state.m.tensor, state.tensor([0., 0.5]))
        loss = torch.mean(torch.abs(m_pred[-1] - m_target))
        #print(m_target.mean(dim = (0,1,2)))
        #print(m_pred[-1].mean(dim = (0,1,2)).clone().detach().cpu())
        #print(state.phi.clone().detach().cpu())
        #print(state.theta.clone().detach().cpu())

        loss.backward()
        optimizer.step()

        values = tuple([epoch] + [x.clone().detach().cpu().item() for x in (state.phi, state.theta, loss)])

        f.write("%d %g %g %g\n" % values)
        f.flush()

