import numpy as np
import torch
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
state.material.alpha = 0.1

state.m = nm.VectorFunction(state).fill((0, 0, 1))

# setup external field depending on phi and theta
Hc = 2 * 1e5 / (constants.mu_0 * 8e5)
state.phi = np.pi / 2
state.theta = np.pi / 2
h_ext = lambda phi, theta: torch.stack(
    [
        Hc / 2 * torch.sin(theta) * torch.cos(phi),
        Hc / 2 * torch.sin(theta) * torch.sin(phi),
        Hc / 2 * torch.cos(theta),
    ]
)

# register effective field
nm.ExchangeField().register(state, "exchange")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.ExternalField(h_ext).register(state, "external")
nm.TotalField("exchange", "aniso", "external").register(state)

# set up solver, loss function, etc
llg = nm.LLGSolver(state, parameters=["phi", "theta"])
optimizer = torch.optim.Adam(llg.parameters(), lr=0.05)
my_loss = torch.nn.L1Loss()

m_target = nm.VectorFunction(state).fill((np.sqrt(0.5), 0, np.sqrt(0.5))).tensor

logger = nm.ScalarLogger("log.dat", ["epoch", "phi", "theta", "loss"])
for epoch in range(100):
    print(f"epoch: {epoch}")
    state.epoch = epoch

    optimizer.zero_grad()
    m_pred = llg.solve(state.tensor([0.0, 0.05e-9]))
    state.loss = my_loss(m_pred[-1], m_target)
    state.loss.backward()
    optimizer.step()

    logger.log(state)
