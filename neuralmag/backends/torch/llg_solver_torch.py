# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        logging.info_green("[LLGSolverTorch] Initialize RHS function")

        internal_args = ["t", "m"]
        for param in self._parameters.keys():
            internal_args.append(param)

        self._func, self._args = self._state.get_func(llg_rhs, internal_args)

        for i, param in enumerate(self._parameters.keys()):
            self._args[2 + i] = self._parameters[param]

    def forward(self, t, m):
        return self._scale_t * self._func(t * self._scale_t, m, *self._args[2:])

    def relax(self, tol=2e7 * torch.pi):
        """
        Use time integration of the damping term to relax the magnetization into an
        energetic equilibrium. The convergence criterion is defined in terms of
        the maximum norm of dm/dt in rad/s.

        :param tol: The stopping criterion in rad/s, defaults to 2 pi / 100 ns
        :type tol: float
        """
        dt = 1e-11
        alpha = self._state.tensor(1.0)

        func, args = self._state.get_func(llg_rhs, ["t", "m", "material__alpha"])
        rhs = lambda t, m: self._scale_t * func(t * self._scale_t, m, alpha, *args[3:])

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
