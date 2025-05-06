# SPDX-License-Identifier: MIT

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
