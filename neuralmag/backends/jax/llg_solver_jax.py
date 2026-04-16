# SPDX-License-Identifier: MIT

import diffrax as dfx
import equinox as eqx
import lineax as lx
import optimistix as optx

import jax
import jax.numpy as jnp
from neuralmag.common import logging

__all__ = ["LLGSolverJAX"]


# monkey patch vector field in MultiTerm
def vector_field(self, *args, **kwargs):
    return sum([term.vector_field(*args, **kwargs) for term in self.terms])


dfx.MultiTerm.vector_field = vector_field


def llg_rhs(h, m, material__alpha):
    gamma_prime = 221276.14725379366 / (1.0 + material__alpha**2)
    return -gamma_prime * jnp.cross(m, h) - material__alpha * gamma_prime * jnp.cross(m, jnp.cross(m, h))


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
    :param max_steps: maximum internal steps to be taken in LLGSolver.step
    :type max_steps: int
    :param solver_type: Solver type for time integration (Dopri5,Kvaerno3,Kvaerno5,KenCarp3,KenCarp5)
    :type solver_type: str
    :param rtol: relative tolerance of time integrator
    :type rtol: float
    :param atol: absolute tolerance of time integrator
    :type atol: float

    :Required state attributes:
        * **state.t** (*scalar*) The time in s
        * **state.h** (*nodal vector field*) The effective field in A/m (in case of implicit/explict solvers)
        * **state.h_impl** (*nodal vector field*) The effective field in A/m to be integrated implicitly (in case of IMEX solvers)
        * **state.h_expl** (*nodal vector field*) The effective field in A/m to be integrated explicitly (in case of IMEX solvers)
        * **state.m** (*nodal vector field*) The magnetization
    """

    def __init__(
        self,
        state,
        scale_t=1e-9,
        parameters=None,
        max_steps=4096,
        solver_type="Dopri5",
        rtol=1e-5,
        atol=1e-5,
    ):
        super().__init__()
        self._state = state
        self._scale_t = scale_t
        self._parameters = [] if parameters is None else parameters
        self._dt0 = 1e-14

        # solver options
        self._max_steps = max_steps

        # TODO Solver options
        solver_types = {
            "Dopri5": ("explicit", dfx.Dopri5),
            "Kvaerno3": ("implicit", dfx.Kvaerno3),
            "Kvaerno5": ("implicit", dfx.Kvaerno5),
            "KenCarp3": ("imex", dfx.KenCarp3),
            "KenCarp5": ("imex", dfx.KenCarp5),
        }
        self._solver_type, self._solver_class = solver_types[solver_type]

        self._stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
        self._saveat_step = dfx.SaveAt(t1=True, dense=False, steps=False)

        self.reset()

    def _setup_term(self, replace={}):
        internal_args = ["t", "m"] + list(replace.keys()) + self._parameters

        if self._solver_type == "explicit" or self._solver_type == "implicit":
            llg_rhs_resolved = self._state.resolve(llg_rhs, internal_args)

            def llg_rhs_scaled(t, m, args):
                return self._scale_t * llg_rhs_resolved(t * self._scale_t, m, *replace.values(), *args)

            return dfx.ODETerm(jax.jit(llg_rhs_scaled))

        if self._solver_type == "imex":
            llg_rhs_resolved_impl = self._state.resolve(llg_rhs, internal_args, remap={"h": "h_impl"})

            def llg_rhs_scaled_impl(t, m, args):
                return self._scale_t * llg_rhs_resolved_impl(t * self._scale_t, m, *replace.values(), *args)

            term_impl = dfx.ODETerm(jax.jit(llg_rhs_scaled_impl))

            llg_rhs_resolved_expl = self._state.resolve(llg_rhs, internal_args, remap={"h": "h_expl"})

            def llg_rhs_scaled_expl(t, m, args):
                return self._scale_t * llg_rhs_resolved_expl(t * self._scale_t, m, *replace.values(), *args)

            term_expl = dfx.ODETerm(jax.jit(llg_rhs_scaled_expl))

            return dfx.MultiTerm(term_expl, term_impl)

    def reset(self):
        """
        Set up the function for the RHS evaluation of the LLG
        """
        logging.info_green("[LLGSolverJAX] Initialize RHS function")

        self._term = self._setup_term()

        if self._solver_type == "explicit":
            self._solver = self._solver_class()
        else:
            root_finder = dfx.with_stepsize_controller_tols(optx.Newton)(
                linear_solver=lx.GMRES(rtol=1e-3, atol=1e-3, restart=50)
            )
            self._solver = self._solver_class(root_finder=root_finder)

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
        term = self._setup_term({"material__alpha": alpha})

        logging.info_blue(f"[LLGSolverJAX] Relaxation started, initial energy E = {self._state.E:g} J")
        t = self._scale_t * self._state.t
        args = []  # TODO is that right?
        while jnp.linalg.norm(term.vector_field(t, self._state.m.tensor, args), axis=-1).max() / self._scale_t > tol:
            logging.info_blue(
                f"[LLGSolverJAX] Relaxation step (max dm/dt = {jnp.linalg.norm(term.vector_field(t, self._state.m.tensor, args), axis=-1).max() / self._scale_t:g}) 1/s"
            )
            sol = dfx.diffeqsolve(
                term,
                self._solver,
                t0=self._state.t / self._scale_t,
                t1=(self._state.t + dt) / self._scale_t,
                dt0=self._dt0 / self._scale_t,
                y0=self._state.m.tensor,
                args=args,
                saveat=self._saveat_step,
                stepsize_controller=self._stepsize_controller,
                max_steps=self._max_steps,
            )
            self._state.m.tensor = sol.ys[-1]

        logging.info_blue(f"[LLGSolverJAX] Relaxation finished, final energy E = {self._state.E:g} J")

    def step(self, dt, *args):
        """
        Perform single integration step of LLG. Internally an adaptive time step is
        used.

        :param dt: The size of the time step
        :type dt: float
        TODO args
        """
        logging.info_blue(f"[LLGSolverJAX] Step: dt = {dt:g}s, t = {self._state.t:g}s")

        sol = dfx.diffeqsolve(
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
        saveat = dfx.SaveAt(ts=t_scaled)
        sol = dfx.diffeqsolve(
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
