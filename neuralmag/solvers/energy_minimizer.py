# SPDX-License-Identifier: MIT

from neuralmag.common import config

__all__ = ["EnergyMinimizer"]


def EnergyMinimizer(state, **kwargs):
    """
    Factory method that returns a backend specific steepest-descent energy
    minimizer.

    :param state: The state used for the minimization
    :type state: :class:`State`
    :param **kwargs: Additional backend specific minimizer options
    :type **kwargs: dict

    :Required state attributes:
        * **state.h** (*nodal vector field*) The effective field in A/m
        * **state.m** (*nodal vector field*) The magnetization

    """

    return config.backend.EnergyMinimizer(state, **kwargs)