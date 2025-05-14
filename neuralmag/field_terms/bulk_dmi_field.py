# SPDX-License-Identifier: MIT

from sympy.vector import curl

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

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
