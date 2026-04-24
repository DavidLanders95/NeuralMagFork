# SPDX-License-Identifier: MIT

import torch

from neuralmag.common import logging

__all__ = ["EnergyMinimizerTorch"]


class EnergyMinimizerTorch:
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

    @property
    def n_iter(self):
        return self._iteration

    @staticmethod
    def _scalar_float(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    @classmethod
    def _log_minimization_status(cls, label, iteration, max_iter, max_g, tol):
        max_g_value = cls._scalar_float(max_g)
        tol_value = cls._scalar_float(tol)
        converged = max_g_value <= tol_value
        logging.info_blue(
            f"[EnergyMinimizerTorch] {label} {iteration}/{max_iter}: "
            f"max_g = {max_g_value:g} (tol = {tol_value:g}, converged = {converged})"
        )
        return converged

    def _effective_field(self, m):
        self._state.m.tensor[:] = m
        return self._state.h.tensor

    @staticmethod
    def _descent_direction(m, h):
        return torch.linalg.cross(m, torch.linalg.cross(m, h))

    @staticmethod
    def _flatten_dot(lhs, rhs):
        return torch.sum(lhs.reshape(-1) * rhs.reshape(-1))

    def _initial_tau(self, h):
        if self._tau_init is not None:
            tau = self._tau_init
        else:
            h_max = torch.linalg.norm(h, dim=-1).max()
            tau = 1.0 / torch.clamp(h_max, min=self._tau_min)
        return float(torch.clamp(self._state.tensor(tau), min=self._tau_min, max=self._tau_max))

    def _bb_tau(self, m, g, h):
        if self._m_prev is None or self._g_prev is None:
            return self._initial_tau(h)

        s = m - self._m_prev
        y = g - self._g_prev
        sy = self._flatten_dot(s, y)
        yy = self._flatten_dot(y, y)
        ss = self._flatten_dot(s, s)

        if self._method == "bb1" or (self._method == "alternating" and self._iteration % 2 == 1):
            denom = sy
            numer = ss
        else:
            denom = yy
            numer = sy

        if not torch.isfinite(denom) or not torch.isfinite(numer) or denom <= 0 or numer <= 0:
            return self._tau if self._tau is not None else self._initial_tau(h)

        tau = numer / denom
        return float(torch.clamp(tau, min=self._tau_min, max=self._tau_max))

    @staticmethod
    def _cayley_update(m, h, tau):
        a = -tau * torch.linalg.cross(m, h)
        a2 = torch.sum(a * a, dim=-1, keepdim=True)
        return ((1.0 - 0.25 * a2) * m - torch.linalg.cross(a, m)) / (1.0 + 0.25 * a2)

    @staticmethod
    def _projected_update(m, g, tau):
        m_new = m - tau * g
        return m_new / torch.clamp(torch.linalg.norm(m_new, dim=-1, keepdim=True), min=torch.finfo(m.dtype).eps)

    def step(self):
        m = self._state.m.tensor.clone()
        h = self._state.h.tensor
        g = self._descent_direction(m, h)

        self._tau = self._bb_tau(m, g, h)
        tau_tensor = self._state.tensor(self._tau)

        if self._update == "projected":
            m_new = self._projected_update(m, g, tau_tensor)
        else:
            m_new = self._cayley_update(m, h, tau_tensor)

        self._m_prev = m
        self._g_prev = g
        self._state.m.tensor[:] = m_new
        self._iteration += 1

        h_new = self._state.h.tensor
        g_new = self._descent_direction(self._state.m.tensor, h_new)
        return torch.linalg.norm(g_new, dim=-1).max()

    def minimize(self, tol=None, max_iter=None, logger=None):
        tol = self._tol if tol is None else tol
        max_iter = self._max_iter if max_iter is None else max_iter

        logging.info_blue(f"[EnergyMinimizerTorch] Minimization started, initial energy E = {self._state.E:g} J")

        max_g = torch.linalg.norm(self._descent_direction(self._state.m.tensor, self._state.h.tensor), dim=-1).max()
        converged = self._log_minimization_status("Initial state", self._iteration, max_iter, max_g, tol)
        while self._iteration < max_iter and not converged:
            max_g = self.step()
            if logger is not None:
                logger.log(self._state)
            converged = self._log_minimization_status("Step", self._iteration, max_iter, max_g, tol)

        logging.info_blue(
            f"[EnergyMinimizerTorch] Minimization finished after {self._iteration} steps, "
            f"final energy E = {self._state.E:g} J, converged = {converged}"
        )

        return max_g

    def relax(self, tol=None, max_iter=None, logger=None):
        return self.minimize(tol=tol, max_iter=max_iter, logger=logger)
