from ..common import logging, constants
from .rkf45 import RKF45
import torch

__all__ = ["LLGSolver"]

class LLGSolver(object):
    def __init__(self, terms, atol = 1e-5):
        self._terms = terms
        def dm(state, t, m):
            t0 = state.t
            m0 = state.m.tensor.detach()
            state.t = t
            state.m.tensor[:] = m
            dm = self._dm(state)
            state.t = t0
            state.m.tensor[:] = m0
            return dm
        self._solver = RKF45(dm, atol=atol)

    def _dm(self, state):
        gamma_prime = constants.gamma / (1. + state.material.alpha**2)
        alpha_prime = state.material.alpha * gamma_prime

        h = sum([term.h(state).tensor for term in self._terms])
        return - gamma_prime * torch.cross(state.m.tensor, h) \
               - alpha_prime * torch.cross(state.m.tensor, torch.cross(state.m.tensor, h))

    def step(self, state, dt):
        self._solver.step(state, dt)
        logging.info_blue("[LLG] step: dt= %g  t=%g" % (dt, state.t))

    #def relax(self, state, maxiter = 500, rtol = 1e-5, dt = 1e-11):
    #    alpha0 = state.material.alpha
    #    t0 = state.t
    #    state.material.alpha = 1.0
    #    E0 = sum([term.E(state) for term in self._terms])
    #    for i in range(maxiter) :
    #        self._solver.step(state, dt)
    #        E = sum([term.E(state) for term in self._terms])
    #        dE = torch.linalg.norm(((E - E0)/E).reshape(-1), ord = float("Inf"))
    #        logging.info_blue("[LLG] relax: t=%g dE=%g E=%g" % (state.t-t0, dE, E))
    #        if dE < rtol:
    #            break
    #        E0 = E
    #    state.t = t0
    #    state.material["alpha"] = alpha0
