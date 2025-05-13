# SPDX-License-Identifier: MIT

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

    def __init__(self, state, scale_t=1e-9, parameters=None, max_steps=4096):
        super().__init__()
        self._state = state
        self._scale_t = scale_t
        self._parameters = [] if parameters is None else parameters
        self._dt0 = 1e-14

        # solver options
        self._max_steps = max_steps

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
        logging.info_green("[LLGSolverJAX] Initialize RHS function")

        internal_args = ["t", "m"] + self._parameters

        self._func = self._state.resolve(llg_rhs, internal_args)
        rhs = lambda t, m, args: self._scale_t * self._func(t * self._scale_t, m, *args)
        self._term = ODETerm(jax.jit(rhs))
        self._solver_state = None

    def relax(self, tol=2e7 * jnp.pi, dt=1e-11):
        """
        Use time integration of the damping term to relax the magnetization into an
        energetic equilibrium. The convergence criterion is defined in terms of
        the maximum norm of dm/dt in rad/s.

        :param tol: The stopping criterion in rad/s, defaults to 2 pi / 100 ns
        :type tol: float
        :param dt: Interval for checking convergence
        :type dt: float
        """
        alpha = self._state.tensor(1.0)

        func = self._state.resolve(llg_rhs, ["t", "m", "material__alpha"])
        rhs = lambda t, m, _: self._scale_t * func(t * self._scale_t, m, alpha)
        term = ODETerm(jax.jit(rhs))

        logging.info_blue(
            f"[LLGSolverJAX] Relaxation started, initial energy E = {self._state.E:g} J"
        )
        t = self._scale_t * self._state.t
        # rhs_args = [self._scale_t * self._state.t, self._state.m.tensor, None]
        while (
            jnp.linalg.norm(
                term.vector_field(t, self._state.m.tensor, None), axis=-1
            ).max()
            / self._scale_t
            > tol
        ):
            logging.info_blue(
                f"[LLGSolverJAX] Relaxation step (max dm/dt = {jnp.linalg.norm(term.vector_field(t, self._state.m.tensor, None), axis=-1).max() / self._scale_t:g}) 1/s"
            )
            sol = diffeqsolve(
                term,
                self._solver,
                t0=self._state.t / self._scale_t,
                t1=(self._state.t + dt) / self._scale_t,
                dt0=self._dt0 / self._scale_t,
                y0=self._state.m.tensor,
                saveat=self._saveat_step,
                stepsize_controller=self._stepsize_controller,
                max_steps=self._max_steps,
            )
            self._state.m.tensor = sol.ys[-1]

        logging.info_blue(
            f"[LLGSolverJAX] Relaxation finished, final energy E = {self._state.E:g} J"
        )

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
            max_steps=self._max_steps,
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
            max_steps=self._max_steps,
        )
        return sol
