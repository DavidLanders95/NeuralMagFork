# SPDX-License-Identifier: MIT

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

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
