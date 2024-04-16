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
    """
    _name = "idmi"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Di", "c" * dim)
        axis = Variable("material__Di_axis", "c" * dim, (3,))
        return (
            D * (m.dot(gradient(m.dot(axis))) - divergence(m) * m.dot(axis)) * dV(dim)
        )
