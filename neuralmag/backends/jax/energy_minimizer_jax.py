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

    :param state: The state used for the minimization
    :type state: :class:`State`
    :param method: BB step-size variant ("bb1", "bb2", or "alternating")
    :type method: str
    :param update: Magnetization update rule ("cayley" or "projected")
    :type update: str
    :param tau_init: Optional initial step size. If None, derive from the
        maximum effective field magnitude.
    :type tau_init: float, optional
    :param tau_min: Lower clamp for the BB step size
    :type tau_min: float
    :param tau_max: Upper clamp for the BB step size
    :type tau_max: float
    :param tol: Convergence threshold on max ||m x (m x h)||
    :type tol: float
    :param max_iter: Maximum number of minimization steps
    :type max_iter: int
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
        valid_methods = {"bb1", "bb2", "alternating"}
        if method not in valid_methods:
            raise ValueError(f"Unsupported BB method {method!r}. Expected one of {sorted(valid_methods)}.")

        valid_updates = {"cayley", "projected"}
        if update not in valid_updates:
            raise ValueError(f"Unsupported update rule {update!r}. Expected one of {sorted(valid_updates)}.")

        self._state = state
        self._method = method
        self._update = update
        self._use_projected_update = update == "projected"
        self._tau_init = tau_init
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._tol = tol
        self._max_iter = max_iter

        self.reset()

        self._effective_field = jax.jit(self._state.resolve(effective_field_fn, ["m"]))
        self._batched_effective_field = jax.jit(jax.vmap(self._effective_field))
        self._compiled_minimize = jax.jit(self._minimize_impl)
        self._compiled_minimize_many = jax.jit(self._minimize_many_impl)

    @property
    def n_iter(self):
        return self._iteration

    def reset(self):
        """
        Reset the Barzilai-Borwein history accumulated by previous steps.
        """
        self._iteration = 0
        self._m_prev = None
        self._g_prev = None
        self._tau = None

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
        tau_prev = self._tau if self._tau is not None else self._initial_tau(h)
        if self._m_prev is None or self._g_prev is None:
            return tau_prev

        return self._bb_tau_single(m, g, self._m_prev, self._g_prev, tau_prev, self._iteration, True)

    @staticmethod
    def _cayley_update(m, h, tau):
        a = -EnergyMinimizerJAX._expand_tau(tau, m) * jnp.cross(m, h)
        a2 = jnp.sum(a * a, axis=-1, keepdims=True)
        return ((1.0 - 0.25 * a2) * m - jnp.cross(a, m)) / (1.0 + 0.25 * a2)

    @staticmethod
    def _projected_update(m, g, tau):
        m_new = m - EnergyMinimizerJAX._expand_tau(tau, m) * g
        return m_new / jnp.maximum(jnp.linalg.norm(m_new, axis=-1, keepdims=True), jnp.finfo(m.dtype).eps)

    def _bb_tau_single(self, m, g, m_prev, g_prev, tau_prev, iteration, has_history):
        s = m - m_prev
        y = g - g_prev
        sy = self._flatten_dot(s, y)
        yy = self._flatten_dot(y, y)
        ss = self._flatten_dot(s, s)

        use_bb1 = self._use_bb1(iteration)
        numer = jnp.where(use_bb1, ss, sy)
        denom = jnp.where(use_bb1, sy, yy)
        valid = has_history & (denom > 0) & (numer > 0)
        tau = jnp.where(valid, numer / denom, tau_prev)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _bb_tau_batch(self, m, g, m_prev, g_prev, tau_prev, iteration, has_history):
        s = m - m_prev
        y = g - g_prev
        sy = self._batch_flatten_dot(s, y)
        yy = self._batch_flatten_dot(y, y)
        ss = self._batch_flatten_dot(s, s)

        use_bb1 = self._use_bb1(iteration)
        numer = jnp.where(use_bb1, ss, sy)
        denom = jnp.where(use_bb1, sy, yy)
        valid = has_history & (denom > 0) & (numer > 0)
        tau = jnp.where(valid, numer / denom, tau_prev)
        return jnp.clip(tau, self._tau_min, self._tau_max)

    def _compute_current_direction(self, m):
        h = self._effective_field(m)
        g = self._descent_direction(m, h)
        max_g = self._max_norm(g)
        return h, g, max_g

    def _step_impl(self, m, h, g, tau):
        if self._use_projected_update:
            m_new = self._projected_update(m, g, tau)
        else:
            m_new = self._cayley_update(m, h, tau)

        return m_new

    def _minimize_impl(self, m, tol, max_iter):
        tol = jnp.asarray(tol, dtype=m.dtype)
        max_iter = jnp.asarray(max_iter, dtype=jnp.int32)

        h0, g0, max_g0 = self._compute_current_direction(m)
        tau0 = self._initial_tau(h0)

        carry = {
            "m": m,
            "h": h0,
            "g": g0,
            "m_prev": m,
            "g_prev": g0,
            "tau": tau0,
            "has_history": jnp.asarray(False),
            "n_iter": jnp.asarray(0, dtype=jnp.int32),
            "max_g": max_g0,
        }

        def cond_fn(inner_carry):
            return (inner_carry["n_iter"] < max_iter) & (inner_carry["max_g"] > tol)

        def body_fn(inner_carry):
            m = inner_carry["m"]
            h = inner_carry["h"]
            g = inner_carry["g"]
            tau = self._bb_tau_single(
                m,
                g,
                inner_carry["m_prev"],
                inner_carry["g_prev"],
                inner_carry["tau"],
                inner_carry["n_iter"],
                inner_carry["has_history"],
            )
            m_candidate = self._step_impl(m, h, g, tau)
            h_candidate, g_candidate, max_g_candidate = self._compute_current_direction(m_candidate)

            return {
                "m": m_candidate,
                "h": h_candidate,
                "g": g_candidate,
                "m_prev": m,
                "g_prev": g,
                "tau": tau,
                "has_history": jnp.asarray(True),
                "n_iter": inner_carry["n_iter"] + jnp.asarray(1, dtype=jnp.int32),
                "max_g": max_g_candidate,
            }

        result = jax.lax.while_loop(cond_fn, body_fn, carry)
        return (
            result["m"],
            result["n_iter"],
            result["max_g"] <= tol,
            result["max_g"],
            result["m_prev"],
            result["g_prev"],
            result["tau"],
        )

    def _minimize_many_impl(self, m_batch, tol, max_iter):
        tol = jnp.asarray(tol, dtype=m_batch.dtype)
        max_iter = jnp.asarray(max_iter, dtype=jnp.int32)

        h0 = self._batched_effective_field(m_batch)
        g0 = self._descent_direction(m_batch, h0)
        max_g0 = self._batch_max_norm(g0)
        tau0 = self._initial_tau_batch(h0)

        carry = {
            "m": m_batch,
            "h": h0,
            "g": g0,
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
            h = inner_carry["h"]
            g = inner_carry["g"]
            tau = self._bb_tau_batch(
                m,
                g,
                inner_carry["m_prev"],
                inner_carry["g_prev"],
                inner_carry["tau"],
                inner_carry["n_iter"],
                inner_carry["has_history"],
            )

            m_candidate = self._step_impl(m, h, g, tau)
            h_candidate = self._batched_effective_field(m_candidate)
            g_candidate = self._descent_direction(m_candidate, h_candidate)
            max_g_candidate = self._batch_max_norm(g_candidate)
            active_mask = self._expand_mask(inner_carry["active"], m)

            return {
                "m": jnp.where(active_mask, m_candidate, m),
                "h": jnp.where(active_mask, h_candidate, h),
                "g": jnp.where(active_mask, g_candidate, g),
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

    @staticmethod
    def _build_info(n_iter, converged, max_g):
        return {"n_iter": n_iter, "converged": converged, "max_g": max_g}

    def step(self):
        m = self._state.m.tensor
        h, g, _ = self._compute_current_direction(m)
        tau = self._bb_tau(m, g, h)
        m_new = self._step_impl(m, h, g, tau)
        _, _, max_g = self._compute_current_direction(m_new)

        self._m_prev = m
        self._g_prev = g
        self._tau = tau
        self._state.m.tensor = m_new
        self._iteration += 1

        return max_g

    def minimize(self, tol=None, max_iter=None, logger=None, return_info=False):
        tol = self._tol if tol is None else tol
        max_iter = self._max_iter if max_iter is None else max_iter
        remaining_iter = max(0, int(max_iter) - self._iteration)

        logging.info_blue(f"[EnergyMinimizerJAX] Minimization started, initial energy E = {self._state.E:g} J")

        if logger is None:
            m_final, n_iter, converged, max_g, m_prev, g_prev, tau = self._compiled_minimize(
                self._state.m.tensor, tol, remaining_iter
            )
            self._state.m.tensor = m_final

            n_iter = int(n_iter)
            if n_iter > 0:
                self._m_prev = m_prev
                self._g_prev = g_prev
                self._tau = tau
                self._iteration += n_iter
        else:
            start_iter = self._iteration
            _, _, max_g = self._compute_current_direction(self._state.m.tensor)
            while self._iteration < max_iter and max_g > tol:
                max_g = self.step()
                logger.log(self._state)

            n_iter = self._iteration - start_iter
            converged = max_g <= tol

        logging.info_blue(
            f"[EnergyMinimizerJAX] Minimization finished after {self._iteration} steps, final energy E = {self._state.E:g} J"
        )

        if return_info:
            return max_g, self._build_info(n_iter, converged, max_g)

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
        m_final, n_iter, converged, max_g = self._compiled_minimize_many(m_batch, tol, max_iter)

        if not return_info:
            return m_final

        return m_final, self._build_info(n_iter, converged, max_g)

    def relax(self, tol=None, max_iter=None, logger=None, return_info=False):
        return self.minimize(tol=tol, max_iter=max_iter, logger=logger, return_info=return_info)
