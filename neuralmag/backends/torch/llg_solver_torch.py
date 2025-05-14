# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from neuralmag.common import Function, logging

__all__ = ["LLGSolverTorch"]


def llg_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -gamma_prime * torch.linalg.cross(
        m, h
    ) - material__alpha * gamma_prime * torch.linalg.cross(m, torch.linalg.cross(m, h))


class LLGSolverTorch(nn.Module):
    """
    Time integrator using explicit adaptive time-stepping provided by the
    torchdiffeq library (https://github.com/rtqichen/torchdiffeq).

    :param state: The state used for the simulation
    :type state: :class:`State`
    :param scale_t: Internal scaling of time to improve numerical behavior
    :type scale_t: float, optional
    :param parameters: List a attribute names for the adjoint gradient computation.
                       Only required for optimization problems.
    :type parameters: list
    :param solver_options: Solver options passed to torchdiffeq
    :type solver_options: dict

    :Required state attributes:
        * **state.t** (*scalar*) The time in s
        * **state.h** (*nodal vector field*) The effective field in A/m
        * **state.m** (*nodal vector field*) The magnetization

    :Example:
        .. code-block::

            # create state with time and magnetization
            state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
            state.t = 0.0
            state.m = nm.VectorFunction(state).fill((1, 0, 0))

            # register constant Zeeman field as state.h
            nm.ExternalField(torch.Tensor((0, 0, 8e5))).register(state, "")

            # initiali LLGSolver
            llg = LLGSolver(state)

            # perform integration step
            llg.step(1e-12)

    """

    def __init__(
        self, state, scale_t=1e-9, parameters=None, max_steps=None, solver_options=None
    ):
        super().__init__()
        self._state = state
        self._scale_t = scale_t
        self._parameter_names = parameters or []
        self._solver_options = {"method": "dopri5", "atol": 1e-5, "rtol": 1e-5}
        self._solver_options.update(solver_options or {})

        # solver options
        self._max_steps = max_steps  # ignored for now

        self.reset()

    def reset(self):
        """
        Set up the function for the RHS evaluation of the LLG
        """
        logging.info_green("[LLGSolverTorch] Initialize RHS function")

        internal_args = ["t", "m"] + self._parameter_names

        self._parameter_values = []
        for parameter_name in self._parameter_names:
            value = self._state.getattr(parameter_name)
            if isinstance(value, Function):
                value = value.tensor

            parameter = torch.nn.Parameter(value)
            self.register_parameter(parameter_name, parameter)
            self._parameter_values.append(parameter)

        self._func = self._state.resolve(llg_rhs, internal_args)

    def forward(self, t, m):
        return self._scale_t * self._func(t * self._scale_t, m, *self._parameter_values)

    def relax(self, tol=2e7 * torch.pi, dt=1e-11):
        """
        Use time integration of the damping term to relax the magnetization into an
        energetic equilibrium. The convergence criterion is defined in terms of
        the maximum norm of dm/dt in rad/s.

        :param tol: The stopping criterion in rad/s, defaults to 2 pi / 100 ns
        :type tol: float
        :param dt: Interval for checking convergence
        :type dt: float
        """
        dt = 1e-11
        alpha = self._state.tensor(1.0)

        func = self._state.resolve(llg_rhs, ["t", "m", "material__alpha"])
        rhs = lambda t, m: self._scale_t * func(t * self._scale_t, m, alpha)

        logging.info_blue(
            f"[LLGSolverTorch] Relaxation started, initial energy E = {self._state.E:g} J"
        )

        t = self._state.tensor(
            [self._state.t / self._scale_t, (self._state.t + dt) / self._scale_t]
        )
        while rhs(t[0], self._state.m.tensor).norm(dim=-1).max() / self._scale_t > tol:
            logging.info_blue(
                f"[LLGSolverTorch] Relaxation step (max dm/dt = {rhs(t[0], self._state.m.tensor).norm(dim=-1).max() / self._scale_t:g}) 1/s"
            )
            m_next = odeint(
                rhs, self._state.m.tensor, t, adjoint_params=(), **self._solver_options
            )
            self._state.m.tensor[:] = m_next[-1]

        logging.info_blue(
            f"[LLGSolverTorch] Relaxation finished, final energy E = {self._state.E:g} J"
        )

    def step(self, dt):
        """
        Perform single integration step of LLG. Internally an adaptive time step is
        used.

        :param dt: The size of the time step
        :type dt: float
        """
        logging.info_blue(
            f"[LLGSolverTorch] Step: dt = {dt:g}s, t = {self._state.t:g}s"
        )
        t = self._state.tensor(
            [self._state.t / self._scale_t, (self._state.t + dt) / self._scale_t]
        )
        m_next = odeint(self, self._state.m.tensor, t, **self._solver_options)
        self._state.t.fill_(t[-1] * self._scale_t)
        self._state.m.tensor[:] = m_next[-1]

    def solve(self, t):
        """
        Solves the LLG for a list of target times. This routine is specifically
        meant to be used in the context of time-dependent optimization with
        objective functions depending on multiple mangetization snapshots.

        :param t: List of target times
        :type t: torch.Tensor
        """
        return odeint(
            self, self._state.m.tensor, t / self._scale_t, **self._solver_options
        )
