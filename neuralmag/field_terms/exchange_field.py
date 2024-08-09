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

from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["ExchangeField"]


class ExchangeField(FieldTerm):
    r"""
    Effective field contribution corresponding to the micromagnetic exchange energy

    .. math::

      E = \int_\Omega A \big( \nabla m_x^2 + \nabla m_y^2 + \nabla m_z^2 \big) \dx

    with the exchange constant :math:`A` given in units of :math:`\text{J/m}`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.A** (*cell scalar field*) The exchange constant in J/m
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "exchange"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        A = Variable("material__A", "c" * dim)
        return (
            A
            * (
                m.diff(N.x).dot(m.diff(N.x))
                + m.diff(N.y).dot(m.diff(N.y))
                + m.diff(N.z).dot(m.diff(N.z))
            )
            * dV(dim)
        )
