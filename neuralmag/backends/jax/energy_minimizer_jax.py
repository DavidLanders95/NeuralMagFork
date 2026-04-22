# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp

from neuralmag.common import logging

__all__ = ["EnergyMinimizerJAX"]


def effective_field_fn(m, h):
    return h


class EnergyMinimizerJAX(object):
    """
    Projected steepest-descent energy minimizer using a Barzilai-Borwein step.
    """

    def __init__(
        self,
        state,
        method="alternating",
        update="cayley",
        tau_init=None,
        tau_min=1e-18,
        tau_max=1e-4,
        tol=1e-3,
        max_iter=10000,
    ):
        self._state = state
        self._method = method
        self._update = update
        self._tau_init = tau_init
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._tol = tol
        self._max_iter = max_iter

        self._iteration = 0
        self._m_prev = None
        self._g_prev = None
        self._tau = None

        self._effective_field = jax.jit(self._state.resolve(effective_field_fn, ["m"]))
        self._step_impl = jax.jit(self._step_impl)

    @property
    def n_iter(self):
        return self._iteration

    @staticmethod
    def _descent_direction(m, h):
        return jnp.cross(m, jnp.cross(m, h))

    @staticmethod
    def _flatten_dot(lhs, rhs):
        return jnp.sum(lhs.reshape(-1) * rhs.reshape(-1))

    def _initial_tau(self, h):
        if self._tau_init is not None:
            tau = self._tau_init
        else:
            h_max = jnp.linalg.norm(h, axis=-1).max()
            tau = 1.0 / jnp.maximum(h_max, self._tau_min)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _bb_tau(self, m, g, h):
        if self._m_prev is None or self._g_prev is None:
            return self._initial_tau(h)

        s = m - self._m_prev
        y = g - self._g_prev
        sy = self._flatten_dot(s, y)
        yy = self._flatten_dot(y, y)
        ss = self._flatten_dot(s, s)

        if self._method == "bb1" or (self._method == "alternating" and self._iteration % 2 == 1):
            numer = ss
            denom = sy
        else:
            numer = sy
            denom = yy

        tau = jnp.where((denom > 0) & (numer > 0), numer / denom, self._tau if self._tau is not None else self._initial_tau(h))
        return jnp.clip(tau, self._tau_min, self._tau_max)

    @staticmethod
    def _cayley_update(m, h, tau):
        a = -tau * jnp.cross(m, h)
        a2 = jnp.sum(a * a, axis=-1, keepdims=True)
        return ((1.0 - 0.25 * a2) * m - jnp.cross(a, m)) / (1.0 + 0.25 * a2)

    @staticmethod
    def _projected_update(m, g, tau):
        m_new = m - tau * g
        return m_new / jnp.maximum(jnp.linalg.norm(m_new, axis=-1, keepdims=True), jnp.finfo(m.dtype).eps)

    def _step_impl(self, m, tau):
        h = self._effective_field(m)
        g = self._descent_direction(m, h)
        max_g = jnp.linalg.norm(g, axis=-1).max()

        if self._update == "projected":
            m_new = self._projected_update(m, g, tau)
        else:
            m_new = self._cayley_update(m, h, tau)

        return m_new, g, h, max_g

    def step(self):
        m = self._state.m.tensor
        h = self._effective_field(m)
        g = self._descent_direction(m, h)
        tau = self._bb_tau(m, g, h)
        m_new, _, _, _ = self._step_impl(m, tau)
        h_new = self._effective_field(m_new)
        g_new = self._descent_direction(m_new, h_new)
        max_g = jnp.linalg.norm(g_new, axis=-1).max()

        self._m_prev = m
        self._g_prev = g
        self._tau = tau
        self._state.m.tensor = m_new
        self._iteration += 1

        return max_g

    def minimize(self, tol=None, max_iter=None, logger=None):
        tol = self._tol if tol is None else tol
        max_iter = self._max_iter if max_iter is None else max_iter

        logging.info_blue(f"[EnergyMinimizerJAX] Minimization started, initial energy E = {self._state.E:g} J")

        max_g = jnp.linalg.norm(self._descent_direction(self._state.m.tensor, self._effective_field(self._state.m.tensor)), axis=-1).max()
        while self._iteration < max_iter and max_g > tol:
            max_g = self.step()
            if logger is not None:
                logger.log(self._state)

        logging.info_blue(
            f"[EnergyMinimizerJAX] Minimization finished after {self._iteration} steps, final energy E = {self._state.E:g} J"
        )

        return max_g

    def relax(self, tol=None, max_iter=None, logger=None):
        return self.minimize(tol=tol, max_iter=max_iter, logger=logger)