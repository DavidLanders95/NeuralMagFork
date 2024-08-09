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

from sympy.vector import divergence, gradient

from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["InterfaceDMIField"]


class InterfaceDMIField(FieldTerm):
    r"""
    Effective field contribution corresponding to the micromagnetic interface-DMI energy

    .. math::

      E = \int_\Omega D \Big[
         \vec{m} \cdot \nabla (\vec{e}_D \cdot \vec{m}) -
         (\nabla \cdot \vec{m}) (\vec{e}_D \cdot \vec{m})
         \Big] \dx

    with the DMI constant :math:`D` given in units of :math:`\text{J/m}^2`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Di** (*cell scalar field*) The DMI constant in J/m^2
        * **state.material.Di_axis** (*cell vector field*) The DMI surface normal as unit vector field
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "idmi"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Di", "c" * dim)
        axis = Variable("material__Di_axis", "c" * dim, (3,))
        return (
            D * (m.dot(gradient(m.dot(axis))) - divergence(m) * m.dot(axis)) * dV(dim)
        )
