import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from ..common import Function, logging
from .llg_solver import LLGSolver

__all__ = ["RelaxSolver"]


def relax_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return (
        -material__alpha * gamma_prime * torch.linalg.cross(m, torch.linalg.cross(m, h))
    )


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

        for i, param in enumerate(self._parameters.keys()):
            self._args[2 + i] = self._parameters[param]

    def minimize(self, tol, timestep=1e-10):
        E_current = self._state.E
        logging.info_blue(f"[RelaxSolver] Initial Energy E = {E_current:g} J")
        dE = 2 * tol  # double tolarance to make sure first step occurs

        while dE > tol:
            E_prev = E_current
            self.step(timestep, logging=False)
            E_current = self._state.E
            dE = E_prev - E_current
            logging.info_blue(
                f"[RelaxSolver]  dE = {dE:g} J, tol = {tol:g} J, E = {E_current:g} J"
            )

        self._state.t = 0
