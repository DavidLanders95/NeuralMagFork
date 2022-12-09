import torch

from ..common import logging

__all__ = ["RKF45"]

# Runge-Kutta-Fehlberg method with stepsize control
class RKF45(object):
    def __init__(self, f, dt=1e-15, atol=1e-5):
        self._f = f
        self._dt = dt  # state._tensor([dt])
        self._order = 4

        # Numerical Recipies 3rd Edition suggests these values:
        self._headroom = 0.9
        self._maxstep = 1e-12  # state._tensor([1e-11])
        self._minscale = 0.2
        self._maxscale = 10.0
        self._atol = atol

    def _try_step(self, state):
        f, m, t, dt = self._f, state.m.tensor, state.t, self._dt
        k1 = dt * f(state, t, m)
        k2 = dt * f(state, t + 1.0 / 4.0 * dt, m + 1.0 / 4.0 * k1)
        k3 = dt * f(state, t + 3.0 / 8.0 * dt, m + 3.0 / 32.0 * k1 + 9.0 / 32.0 * k2)
        k4 = dt * f(
            state,
            t + 12.0 / 13.0 * dt,
            m + 1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3,
        )
        k5 = dt * f(
            state,
            t + 1.0 * dt,
            m
            + 439.0 / 216.0 * k1
            - 8.0 * k2
            + 3680.0 / 513.0 * k3
            - 845.0 / 4104.0 * k4,
        )
        k6 = dt * f(
            state,
            t + 1.0 / 2.0 * dt,
            m
            - 8.0 / 27.0 * k1
            + 2.0 * k2
            - 3544.0 / 2565.0 * k3
            + 1859.0 / 4104.0 * k4
            - 11.0 / 40.0 * k5,
        )

        dm = (
            16.0 / 135.0 * k1
            + 6656.0 / 12825.0 * k3
            + 28561.0 / 56430.0 * k4
            - 9.0 / 50.0 * k5
            + 2.0 / 55.0 * k6
        )
        rk_error = dm - (
            25.0 / 216.0 * k1
            + 1408.0 / 2565.0 * k3
            + 2197.0 / 4104.0 * k4
            - 1.0 / 5.0 * k5
        )
        return m + dm, t + dt, rk_error

    def _optimal_stepsize(self, rk_error):
        norm = torch.linalg.norm(rk_error.flatten() / self._atol, torch.inf)
        if torch.isnan(norm):
            raise RuntimeError("Unexpected error norm= %.5g!" % norm)

        if norm > 1.1:
            # decrease step, no more than factor of 5, but a fraction S more
            # than scaling suggests (for better accuracy)
            r = self._headroom / torch.pow(norm, 1.0 / self._order)
            if r < self._minscale:
                r = self._minscale
        elif norm < 0.5:
            # increase step, but no more than by a factor of 5
            r = self._headroom / torch.pow(norm, 1.0 / (self._order + 1.0))
            if r > self._maxscale:  # increase no more than factor of 5
                r = self._maxscale
            if r < 1.0:  # don't allow any decrease caused by S<1
                r = 1.0
        else:  # no change
            r = 1.0

        dt_opt = self._dt * r
        if dt_opt > self._maxstep:
            dt_opt = self._maxstep
        return dt_opt

    def step(self, state, dt):
        t0, t1 = state.t, state.t + dt
        while state.t < t1:
            _m1, _t1, err = self._try_step(state)
            dt_opt = self._optimal_stepsize(err)
            if self._dt > dt_opt or self._dt > t1 - state.t:
                # step size was too large, retry with optimal stepsize
                self._dt = min(dt_opt, t1 - state.t).detach()
                logging.debug(
                    "REVERT step: %g, new step size: %g, time: %g"
                    % (self._dt, dt_opt, state.t)
                )
            else:
                # accept step, adapt stepsize for next step
                state.m.tensor[:] = _m1
                state.t = _t1
                logging.debug(
                    "ACCEPT step: %g, new step size: %g, time: %g"
                    % (self._dt, dt_opt, state.t)
                )
                self._dt = dt_opt
