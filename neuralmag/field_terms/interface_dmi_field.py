# SPDX-License-Identifier: MIT

from sympy.vector import divergence, gradient

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

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
