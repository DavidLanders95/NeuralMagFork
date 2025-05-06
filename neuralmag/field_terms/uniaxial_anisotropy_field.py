# SPDX-License-Identifier: MIT

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

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
