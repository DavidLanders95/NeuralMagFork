import torch
from scipy import constants
from ..common import logging
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

__all__ = ["LLGSolver"]

class LLGSolver(nn.Module):
    def __init__(self, state, rtol = 1e-5, atol=1e-5):
        super().__init__()
        self._state = state
        self._rtol = rtol
        self._atop = atol
            
        gamma = constants.physical_constants['electron gyromag. ratio'][0] * constants.mu_0
        self.g_prime = gamma / (1. + state.material.alpha**2)
        self.a_prime = state.material.alpha * self.g_prime

    def  forward(self, t, m):
        # TODO update state.t
        self._state.m.tensor[:] = m[:]
        h = self._state.h_total.tensor
        return - self.g_prime * torch.cross(m, h) - self.a_prime * torch.cross(m, torch.cross(m, h))

    def step(self, dt):
        # TODO scale t be 1e-9??
        logging.info_blue("[LLG] step: dt= %g  t=%g" % (dt, self._state.t))
        t = torch.linspace(self._state.t, self._state.t + dt, 2, dtype = self._state.dtype, device = self._state.device)
        m_next = odeint(self, self._state.m.tensor.detach().clone(), t, method = 'dopri5', rtol = 1e-5, atol = 1e-5) # TODO really need detach clone?
        self._state.t += dt
        self._state.m.tensor[:] = m_next[1]

