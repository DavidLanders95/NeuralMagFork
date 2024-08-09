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

from sympy.vector import curl

from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["BulkDMIField"]


class BulkDMIField(FieldTerm):
    r"""
    Effective field contribution for the micromagnetic bulk-DMI energy

    .. math::

      E = \int_\Omega D \vec{m} \cdot (\nabla \times \vec{m}) \dx

    with the DMI constant :math:`D` given in units of :math:`\text{J/m}^2`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Db** (*cell scalar field*) The DMI constant in J/m^2
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "bdmi"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Db", "c" * dim)
        return D * m.dot(curl(m)) * dV()
