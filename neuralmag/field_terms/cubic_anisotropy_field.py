# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["CubicAnisotropyField"]


class CubicAnisotropyField(FieldTerm):
    r"""
    Effective field contribution corresponding to the cubic anisotropy energy

    .. math::
      E = - \int_\Omega K \big( (\vec{m} \cdot
           \vec{e}_1)^2(\vec{m} \cdot
           \vec{e}_2)^2 + (\vec{m} \cdot
           \vec{e}_2)^2(\vec{m} \cdot
           \vec{e}_3)^2 + (\vec{m} \cdot
           \vec{e}_1)^2(\vec{m} \cdot
           \vec{e}_3)^2 \big) \dx

    with the anisotropy constant :math:`K` given in units of :math:`\text{J/m}^3`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler (defaults to 1).
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
