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

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["CubicAnisotropyField"]


class CubicAnisotropyField(FieldTerm):
    r"""
    Effective field contribution corresponding to the cubic anisotropy energy

    .. math::
      E = - \int_\Omega K \big( (\mathbf{m} \cdot
           \mathbf{v}_{1})^{2}(\mathbf{m} \cdot
           \mathbf{v}_{2})^{2} + (\mathbf{m} \cdot
           \mathbf{v}_{2})^{2}(\mathbf{m} \cdot
           \mathbf{v}_{3})^{2} + (\mathbf{m} \cdot
           \mathbf{v}_{1})^{2}(\mathbf{m} \cdot
           \mathbf{v}_{3})^{2} \big) \dx

    with the anisotropy constant :math:`K` given in units of :math:`\text{J/m}^3`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Kc** (*cell scalar field*) The anisotropy constant in J/m^3
        * **state.material.Kc_axis1** (*cell vector field*) The first anisotropy axis as unit vector field
        * **state.material.Kc_axis2** (*cell vector field*) The second anisotropy axis as unit vector field
        * **state.material.Kc_axis3** (*cell vector field*) The third anisotropy axis as unit vector field
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m

    **state.material.Kc_axis1**, **state.material.Kc_axis2**, and **state.material.Kc_axis3** should be
    orthogonal unit vectors.
    """
    default_name = "caniso"

    def __init__(self, n_gauss=1, **kwargs):
        super().__init__(n_gauss=n_gauss, **kwargs)

    @staticmethod
    def e_expr(m, dim):
        K = Variable("material__Kc", "c" * dim)
        axis1 = Variable("material__Kc_axis1", "c" * dim, (3,))
        axis2 = Variable("material__Kc_axis2", "c" * dim, (3,))
        axis3 = Variable("material__Kc_axis3", "c" * dim, (3,))
        return (
            -K
            * (
                m.dot(axis1) ** 2 * m.dot(axis2) ** 2
                + m.dot(axis2) ** 2 * m.dot(axis3) ** 2
                + m.dot(axis3) ** 2 * m.dot(axis1) ** 2
            )
            * dV(dim)
        )
