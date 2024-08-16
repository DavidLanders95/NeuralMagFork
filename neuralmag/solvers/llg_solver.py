"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from ..common import Function, logging

__all__ = ["LLGSolver"]


def llg_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -gamma_prime * torch.linalg.cross(
        m, h
    ) - material__alpha * gamma_prime * torch.linalg.cross(m, torch.linalg.cross(m, h))


class LLGSolver(nn.Module):
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

    def __init__(self, state, scale_t=1e-9, parameters=None, solver_options=None):
        super().__init__()
        self._state = state
        self._scale_t = scale_t
        self._parameters = {}
        self._solver_options = {"method": "dopri5", "atol": 1e-5, "rtol": 1e-5}
        self._solver_options.update(solver_options or {})
        for param in parameters or []:
            value = state.getattr(param)
            if isinstance(value, Function):
                value = value.tensor
            self._parameters[param] = torch.nn.Parameter(value)

        self.reset()

    def reset(self):
        """
        Set up the function for the RHS evaluation of the LLG
        """
        logging.info_green("[LLGSolver] Initialize RHS function")

        internal_args = ["t", "m"]
        for param in self._parameters.keys():
            internal_args.append(param)

        self._func, self._args = self._state.get_func(llg_rhs, internal_args)

        for i, param in enumerate(self._parameters.keys()):
            self._args[2 + i] = self._parameters[param]

    def forward(self, t, m):
        return self._scale_t * self._func(t * self._scale_t, m, *self._args[2:])

    def step(self, dt):
        """
        Perform single integration step of LLG. Internally an adaptive time step is
        used.

        :param dt: The size of the time step
        :type dt: float
        """
        logging.info_blue(f"[LLGSolver] Step: dt = {dt:g}s, t = {self._state.t:g}s")
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
