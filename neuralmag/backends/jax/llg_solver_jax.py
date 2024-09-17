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

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from neuralmag.common import Function, logging

__all__ = ["LLGSolverJAX"]


def llg_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -gamma_prime * jnp.cross(m, h) - material__alpha * gamma_prime * jnp.cross(
        m, jnp.cross(m, h)
    )


# @jax.jit
@eqx.filter_jit
def solve(term, solver, t, dt0, y0, saveat, stepsize_controller):
    sol = diffeqsolve(
        term,
        solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol


class LLGSolverJAX(object):
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

    :Required state attributes:
        * **state.t** (*scalar*) The time in s
        * **state.h** (*nodal vector field*) The effective field in A/m
        * **state.m** (*nodal vector field*) The magnetization
    """

    def __init__(self, state, scale_t=1e-9, parameters=None):
        super().__init__()
        self._state = state
        self._scale_t = scale_t
        #        self._parameters = {}
        #        self._solver_options = {"method": "dopri5", "atol": 1e-5, "rtol": 1e-5}
        #        self._solver_options.update(solver_options or {})
        #        for param in parameters or []:
        #            value = state.getattr(param)
        #            if isinstance(value, Function):
        #                value = value.tensor
        #            self._parameters[param] = torch.nn.Parameter(value)
        #
        self.reset()

    def reset(self):
        """
        Set up the function for the RHS evaluation of the LLG
        """
        logging.info_green("[LLGSolver] Initialize RHS function")

        internal_args = ["t", "m"]
        #        for param in self._parameters.keys():
        #            internal_args.append(param)

        self._func, self._args = self._state.get_func(llg_rhs, internal_args)
        rhs = lambda t, m, args: self._scale_t * self._func(
            t * self._scale_t, m, *self._args[2:]
        )
        self._rhs = jax.jit(rhs)

    #        for i, param in enumerate(self._parameters.keys()):
    #            self._args[2 + i] = self._parameters[param]

    def relax(self, tol=2e7 * jnp.pi):
        # TODO use proper relaxation
        self.step(1e-9)
        self.reset()

    def step(self, dt):
        """
        Perform single integration step of LLG. Internally an adaptive time step is
        used.

        :param dt: The size of the time step
        :type dt: float
        """
        logging.info_blue(f"[LLGSolverJAX] Step: dt = {dt:g}s, t = {self._state.t:g}s")
        t = self._state.tensor(
            [self._state.t / self._scale_t, (self._state.t + dt) / self._scale_t]
        )
        dt0 = 1e-14 / self._scale_t
        term = ODETerm(self._rhs)
        solver = Dopri5()
        saveat = SaveAt(ts=[t[-1]])
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

        sol = solve(
            term, solver, t, dt0, self._state.m.tensor, saveat, stepsize_controller
        )
        # sol = diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=dt0, y0=self._state.m.tensor, saveat=saveat, stepsize_controller=stepsize_controller)
        self._state.t = sol.ts[-1] * self._scale_t
        self._state.m._tensor = sol.ys[-1]
        return sol


#    def solve(self, t):
#        """
#        Solves the LLG for a list of target times. This routine is specifically
#        meant to be used in the context of time-dependent optimization with
#        objective functions depending on multiple mangetization snapshots.
#
#        :param t: List of target times
#        :type t: torch.Tensor
#        """
#        return odeint(
#            self, self._state.m.tensor, t / self._scale_t, **self._solver_options
#        )
#
#
#        ts = t / self._scale_t
#        dt0 = 1e-14 / self._scale_t
#
#        term = ODETerm(vector_field)
#        solver = Dopri5()
#        saveat = SaveAt(ts=ts)
#        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
#
#        sol = diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=dt0, y0=self._state.m.tensor, saveat=saveat, stepsize_controller=stepsize_controller)
#        return sol
