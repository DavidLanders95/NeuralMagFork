import torch
import torch.nn as nn
from ..common import logging
from torchdiffeq import odeint_adjoint as odeint

__all__ = ["LLGSolver"]

def llg_rhs(h, m, material__alpha):
    gamma_prime = 1e-9 * 221276.14725379366 / (1 + material__alpha**2)
    return - gamma_prime * torch.linalg.cross(m, h) \
           - material__alpha * gamma_prime * torch.linalg.cross(m, torch.linalg.cross(m, h))

class LLGSolver(nn.Module):
    def __init__(self, state, rtol = 1e-5, atol=1e-5, parameters = []):
        super().__init__()
        self._state = state
        self._rtol = rtol
        self._atop = atol

        self._parameters = {}
        for param in parameters:
            # TODO make sure that state.getattr returns a tensor
            self._parameters[param] = torch.nn.Parameter(state.getattr(param))

        self.reset()

    def reset(self):
        logging.info_green("[LLGSolver] Initialize RHS function")

        internal_args = ['t', 'm']
        for param in self._parameters.keys():
            internal_args.append(param)

        self._func, self._args = self._state.get_func(llg_rhs, internal_args)

        for i, param in enumerate(self._parameters.keys()):
            self._args[2 + i] = self._parameters[param]

    def forward(self, t, m):
        return self._func(t * 1e-9, m, *self._args[2:])

    def step(self, dt):
        logging.info_blue("[LLGSolver] Step: dt = %g, t = %g" % (dt, self._state.t))
        t = self._state.tensor([self._state.t * 1e9, (self._state.t + dt) * 1e9])
        m_next = odeint(self, self._state.m.tensor, t, # TODO need to clone m?
                method = 'dopri5', rtol = 1e-5, atol = 1e-5)
        self._state.t.fill_(t[-1] * 1e-9)
        self._state.m.tensor[:] = m_next[-1]
