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

from neuralmag.common import config

__all__ = ["LLGSolver"]


def LLGSolver(state, scale_t=1e-9, parameters=None, **kwargs):
    """
    Factory method that returns a backend specific time-integrator object
    for the LLG (either :class:`LLGSolverTorch` or :class:`LLGSolverJAX`).

    :param state: The state used for the simulation
    :type state: :class:`State`
    :param scale_t: Internal scaling of time to improve numerical behavior
    :type scale_t: float, optional
    :param parameters: List a attribute names for the adjoint gradient computation.
                       Only required for optimization problems.
    :type parameters: list
    :param **kwargs: Additional backend specific solver options
    :type **kwargs: dict

    :Required state attributes:
        * **state.t** (*scalar*) The time in s
        * **state.h** (*nodal vector field*) The effective field in A/m
        * **state.m** (*nodal vector field*) The magnetization

    :Example:
        .. code-block::

            # create state with time and magnetization
            state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
            state.t = 0.0
            state.m = nm.VectorFunction(state).fill((1, 0, 0))

            # register constant Zeeman field as state.h
            nm.ExternalField(torch.Tensor((0, 0, 8e5))).register(state, "")

            # initiali LLGSolver
            llg = LLGSolver(state)

            # perform integration step
            llg.step(1e-12)

    """

    return config.backend.LLGSolver(
        state, scale_t=scale_t, parameters=parameters, **kwargs
    )
