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
        self._batched_effective_field = jax.jit(jax.vmap(self._effective_field))
        self._step_impl = jax.jit(self._step_impl)
        self._minimize_many_impl = jax.jit(self._minimize_many_impl)

    @property
    def n_iter(self):
        return self._iteration

    @staticmethod
    def _descent_direction(m, h):
        return jnp.cross(m, jnp.cross(m, h))

    @staticmethod
    def _flatten_dot(lhs, rhs):
        return jnp.sum(lhs.reshape(-1) * rhs.reshape(-1))

    @staticmethod
    def _batch_flatten_dot(lhs, rhs):
        return jnp.sum(lhs.reshape((lhs.shape[0], -1)) * rhs.reshape((rhs.shape[0], -1)), axis=-1)

    @staticmethod
    def _max_norm(field):
        return jnp.linalg.norm(field, axis=-1).max()

    @staticmethod
    def _batch_max_norm(field):
        return jnp.linalg.norm(field, axis=-1).reshape((field.shape[0], -1)).max(axis=-1)

    @staticmethod
    def _expand_tau(tau, m):
        tau = jnp.asarray(tau, dtype=m.dtype)
        return jnp.reshape(tau, tau.shape + (1,) * max(m.ndim - tau.ndim, 0))

    @staticmethod
    def _expand_mask(mask, value):
        return jnp.reshape(mask, mask.shape + (1,) * max(value.ndim - mask.ndim, 0))

    def _initial_tau(self, h):
        if self._tau_init is not None:
            tau = self._tau_init
        else:
            h_max = self._max_norm(h)
            tau = 1.0 / jnp.maximum(h_max, self._tau_min)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _initial_tau_batch(self, h):
        if self._tau_init is not None:
            tau = jnp.full((h.shape[0],), self._tau_init, dtype=h.dtype)
        else:
            h_max = self._batch_max_norm(h)
            tau = 1.0 / jnp.maximum(h_max, self._tau_min)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _use_bb1(self, iteration):
        if self._method == "bb1":
            return jnp.ones_like(jnp.asarray(iteration), dtype=bool)
        if self._method == "alternating":
            return jnp.asarray(iteration) % 2 == 1
        return jnp.zeros_like(jnp.asarray(iteration), dtype=bool)

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
        a = -EnergyMinimizerJAX._expand_tau(tau, m) * jnp.cross(m, h)
        a2 = jnp.sum(a * a, axis=-1, keepdims=True)
        return ((1.0 - 0.25 * a2) * m - jnp.cross(a, m)) / (1.0 + 0.25 * a2)

    @staticmethod
    def _projected_update(m, g, tau):
        m_new = m - EnergyMinimizerJAX._expand_tau(tau, m) * g
        return m_new / jnp.maximum(jnp.linalg.norm(m_new, axis=-1, keepdims=True), jnp.finfo(m.dtype).eps)

    def _bb_tau_batch(self, m, g, h, m_prev, g_prev, tau_prev, iteration, has_history):
        initial_tau = self._initial_tau_batch(h)
        s = m - m_prev
        y = g - g_prev
        sy = self._batch_flatten_dot(s, y)
        yy = self._batch_flatten_dot(y, y)
        ss = self._batch_flatten_dot(s, s)

        use_bb1 = self._use_bb1(iteration)
        numer = jnp.where(use_bb1, ss, sy)
        denom = jnp.where(use_bb1, sy, yy)
        fallback = jnp.where(has_history, tau_prev, initial_tau)
        valid = has_history & (denom > 0) & (numer > 0)
        tau = jnp.where(valid, numer / denom, fallback)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _step_impl(self, m, tau):
        h = self._effective_field(m)
        g = self._descent_direction(m, h)
        max_g = self._max_norm(g)

        if self._update == "projected":
            m_new = self._projected_update(m, g, tau)
        else:
            m_new = self._cayley_update(m, h, tau)

        return m_new, g, h, max_g

    def _minimize_many_impl(self, m_batch, tol, max_iter):
        tol = jnp.asarray(tol, dtype=m_batch.dtype)
        max_iter = jnp.asarray(max_iter, dtype=jnp.int32)

        h0 = self._batched_effective_field(m_batch)
        g0 = self._descent_direction(m_batch, h0)
        max_g0 = self._batch_max_norm(g0)
        tau0 = self._initial_tau_batch(h0)

        carry = {
            "m": m_batch,
            "m_prev": m_batch,
            "g_prev": g0,
            "tau": tau0,
            "has_history": jnp.zeros_like(max_g0, dtype=bool),
            "active": max_g0 > tol,
            "n_iter": jnp.zeros_like(max_g0, dtype=jnp.int32),
            "max_g": max_g0,
            "loop_iter": jnp.asarray(0, dtype=jnp.int32),
        }

        def cond_fn(inner_carry):
            return (inner_carry["loop_iter"] < max_iter) & jnp.any(inner_carry["active"])

        def body_fn(inner_carry):
            m = inner_carry["m"]
            h = self._batched_effective_field(m)
            g = self._descent_direction(m, h)
            tau = self._bb_tau_batch(
                m,
                g,
                h,
                inner_carry["m_prev"],
                inner_carry["g_prev"],
                inner_carry["tau"],
                inner_carry["n_iter"],
                inner_carry["has_history"],
            )

            if self._update == "projected":
                m_candidate = self._projected_update(m, g, tau)
            else:
                m_candidate = self._cayley_update(m, h, tau)

            h_candidate = self._batched_effective_field(m_candidate)
            g_candidate = self._descent_direction(m_candidate, h_candidate)
            max_g_candidate = self._batch_max_norm(g_candidate)
            active_mask = self._expand_mask(inner_carry["active"], m)

            return {
                "m": jnp.where(active_mask, m_candidate, m),
                "m_prev": jnp.where(active_mask, m, inner_carry["m_prev"]),
                "g_prev": jnp.where(active_mask, g, inner_carry["g_prev"]),
                "tau": jnp.where(inner_carry["active"], tau, inner_carry["tau"]),
                "has_history": inner_carry["has_history"] | inner_carry["active"],
                "active": inner_carry["active"] & (max_g_candidate > tol),
                "n_iter": inner_carry["n_iter"] + inner_carry["active"].astype(jnp.int32),
                "max_g": jnp.where(inner_carry["active"], max_g_candidate, inner_carry["max_g"]),
                "loop_iter": inner_carry["loop_iter"] + jnp.asarray(1, dtype=jnp.int32),
            }

        result = jax.lax.while_loop(cond_fn, body_fn, carry)
        return result["m"], result["n_iter"], ~result["active"], result["max_g"]

    def step(self):
        m = self._state.m.tensor
        h = self._effective_field(m)
        g = self._descent_direction(m, h)
        tau = self._bb_tau(m, g, h)
        m_new, _, _, _ = self._step_impl(m, tau)
        h_new = self._effective_field(m_new)
        g_new = self._descent_direction(m_new, h_new)
        max_g = self._max_norm(g_new)

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

        max_g = self._max_norm(self._descent_direction(self._state.m.tensor, self._effective_field(self._state.m.tensor)))
        while self._iteration < max_iter and max_g > tol:
            max_g = self.step()
            if logger is not None:
                logger.log(self._state)

        logging.info_blue(
            f"[EnergyMinimizerJAX] Minimization finished after {self._iteration} steps, final energy E = {self._state.E:g} J"
        )

        return max_g

    def minimize_many(self, m_batch, tol=None, max_iter=None, return_info=False):
        """
        Minimize a batch of independent magnetization states in parallel.

        This JAX-only helper uses the current solver's bound state as a template
        for material parameters and compiled field terms, but it does not mutate
        ``state.m``. ``m_batch`` must have shape ``(batch, *state.m.tensor.shape)``.
        """

        tol = self._tol if tol is None else tol
        max_iter = self._max_iter if max_iter is None else max_iter

        state_shape = tuple(self._state.m.tensor.shape)
        if tuple(m_batch.shape[1:]) != state_shape:
            raise ValueError(
                f"Expected m_batch with trailing shape {state_shape}, got {tuple(m_batch.shape[1:])}."
            )

        m_batch = jnp.asarray(m_batch, dtype=self._state.m.tensor.dtype)
        m_final, n_iter, converged, max_g = self._minimize_many_impl(m_batch, tol, max_iter)

        if not return_info:
            return m_final

        return m_final, {"n_iter": n_iter, "converged": converged, "max_g": max_g}

    def relax(self, tol=None, max_iter=None, logger=None):
        return self.minimize(tol=tol, max_iter=max_iter, logger=logger)