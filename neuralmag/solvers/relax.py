import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint_event

from ..common import Function, logging
from .llg_solver import LLGSolver

__all__ = ["RelaxSolver"]


def relax_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -material__alpha * gamma_prime * torque(h, m)


def torque(h, m):
    return torch.linalg.cross(m, torch.linalg.cross(m, h))


class RelaxSolver(LLGSolver):
    """
    Minalise the total energy using a time integrator using explicit adaptive time-stepping provided by the
    torchdiffeq library.

    :param state: The state used for the simulation
    :type state: :class:`State`
    :param scale_t: Internal scaling of time to improve numerical behavior
    :type scale_t: float, optional
    :param parameters: List a attribute names for the adjoint gradient computation
    :type parameters: list
    :param solver_options: Solver options passed to torchdiffeq
    :type solver_options: dict
    """

    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)

    def reset(self):
        """
        Set up the function for the RHS evaluation of the Relax
        """
        logging.info_green("[RelaxSolver] Initialize RHS function")

        internal_args = ["t", "m"]
        for param in self._parameters.keys():
            internal_args.append(param)

        self._func, self._args = self._state.get_func(relax_rhs, internal_args)
        self._func_torque, self._args_torque = self._state.get_func(
            torque, internal_args
        )

        for i, param in enumerate(self._parameters.keys()):
            self._args[2 + i] = self._parameters[param]
            self._args_torque[2 + i] = self._parameters[param]

    def minimize(self, tol):
        self.tol = tol
        t0 = self._state.t
        logging.info_blue(f"[RelaxSolver] Initial Energy E = {self._state.E:g} J")
        _, m_next = odeint_event(
            self,
            self._state.m.tensor,
            self._state.t,
            event_fn=self._event_fn,
            odeint_interface=odeint,
            **self._solver_options,
        )
        self._state.m.tensor[:] = m_next[-1]
        logging.info_blue(f"[RelaxSolver] Final Energy E = {self._state.E:g} J")
        self._state.t = t0

    def _event_fn(self, t, m):
        torque = self._func_torque(t * self._scale_t, m, *self._args_torque[2:])
        if torque.abs().max() > self.tol:
            return torch.ones_like(t)
        else:
            return torch.zeros_like(t)
