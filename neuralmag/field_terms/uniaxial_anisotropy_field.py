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

__all__ = ["UniaxialAnisotropyField"]


class UniaxialAnisotropyField(FieldTerm):
    r"""
    Effective field contribution corresponding to the quadratic uniaxial anisotropy energy

    .. math::
      E = - \int_\Omega K \big( \vec{m} \cdot \vec{e}_k \big)^2 \dx

    with the anisotropy constant :math:`K` given in units of :math:`\text{J/m}^3`.
    For higher order anisotropy, use the :class:`UniaxialAnisotropyField2`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Ku** (*cell scalar field*) The anisotropy constant in J/m^3
        * **state.material.Ku_axis** (*cell vector field*) The anisotropy axis as unit vector field
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "uaniso"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        K = Variable("material__Ku", "c" * dim)
        axis = Variable("material__Ku_axis", "c" * dim, (3,))
        return -K * m.dot(axis) ** 2 * dV(dim)
