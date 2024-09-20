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
from diffrax import Dopri5, Event, ODETerm, PIDController, SaveAt, diffeqsolve

from neuralmag.common import Function, logging

__all__ = ["LLGSolverJAX"]


def llg_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -gamma_prime * jnp.cross(m, h) - material__alpha * gamma_prime * jnp.cross(
        m, jnp.cross(m, h)
    )


def llg_relax_rhs(h, m):
    gamma_prime = 221276.14725379366 / 2.0
    return -gamma_prime * jnp.cross(m, jnp.cross(m, h))


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
        self._parameters = [] if parameters is None else parameters
        self._dt0 = 1e-14

        # TODO Solver options
        # self._solver_options = {"method": "dopri5", "atol": 1e-5, "rtol": 1e-5}
        self._solver = Dopri5()
        self._stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        self._saveat_step = SaveAt(t1=True)

        self.reset()

    def reset(self):
        """
        Set up the function for the RHS evaluation of the LLG
        """
        logging.info_green("[LLGSolver] Initialize RHS function")

        internal_args = ["t", "m"] + self._parameters

        self._func, self._args = self._state.get_func(llg_rhs, internal_args)
        rhs = lambda t, m, args: self._scale_t * self._func(
            t * self._scale_t, m, *args, *self._args[len(internal_args) :]
        )
        self._term = ODETerm(jax.jit(rhs))
        self._solver_state = None

    def relax(self, tol=2e7 * jnp.pi):
        """
        Use time integration of the damping term to relax the magnetization into an
        energetic equilibrium. The convergence criterion is defined in terms of
        the maximum norm of dm/dt in rad/s.

        :param tol: The stopping criterion in rad/s, defaults to 2 pi / 100 ns
        :type tol: float
        """
        func, args = self._state.get_func(llg_relax_rhs, ["t", "m"])
        logging.info_blue(
            f"[LLGSolverJAX] Start relaxation, initial energy E = {self._state.E:g} J"
        )
        rhs_fn = lambda t, m, _: self._scale_t * func(
            self._state.t * self._scale_t, m, *args[2:]
        )
        event_fn = (
            lambda t, m, _, **kwargs: jnp.linalg.norm(
                func(self._state.t * self._scale_t, m, *args[2:]), axis=-1
            ).max()
            < tol
        )

        sol = diffeqsolve(
            ODETerm(jax.jit(rhs_fn)),
            self._solver,
            t0=self._state.t / self._scale_t,
            t1=(self._state.t + 1.0) / self._scale_t,
            dt0=self._dt0 / self._scale_t,
            y0=self._state.m.tensor,
            event=Event(eqx.filter_jit(event_fn)),
            stepsize_controller=self._stepsize_controller,
            max_steps=None,
        )
        self._state.m.tensor = sol.ys[-1]
        logging.info_blue(
            f"[LLGSolverJAX] Relaxation finished, final energy E = {self._state.E:g} J"
        )
        return sol

    def step(self, dt, *args):
        """
        Perform single integration step of LLG. Internally an adaptive time step is
        used.

        :param dt: The size of the time step
        :type dt: float
        TODO args
        """
        logging.info_blue(f"[LLGSolverJAX] Step: dt = {dt:g}s, t = {self._state.t:g}s")

        sol = diffeqsolve(
            self._term,
            self._solver,
            t0=self._state.t / self._scale_t,
            t1=(self._state.t + dt) / self._scale_t,
            dt0=self._dt0 / self._scale_t,
            y0=self._state.m.tensor,
            args=args,
            saveat=self._saveat_step,
            stepsize_controller=self._stepsize_controller,
            solver_state=self._solver_state,
        )
        self._solver_state = sol.solver_state
        self._state.t = sol.ts[-1] * self._scale_t
        self._state.m.tensor = sol.ys[-1]
        return sol

    def solve(self, t, *args):
        """
        Solves the LLG for a list of target times. This routine is specifically
        meant to be used in the context of time-dependent optimization with
        objective functions depending on multiple mangetization snapshots.

        :param t: List of target times
        :type t: torch.Tensor
        TODO args
        """
        t_scaled = t / self._scale_t
        saveat = SaveAt(ts=t_scaled)
        sol = diffeqsolve(
            self._term,
            self._solver,
            t0=t_scaled[0],
            t1=t_scaled[-1],
            dt0=self._dt0 / self._scale_t,
            y0=self._state.m.tensor,
            args=args,
            saveat=saveat,
            stepsize_controller=self._stepsize_controller,
        )
        return sol
