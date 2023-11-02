from sympy.vector import divergence, gradient

from ..generators.pytorch_generator import N, Variable
from .field_term import FieldTerm

__all__ = ["InterfaceDMIField"]


class InterfaceDMIField(FieldTerm):
    _name = "idmi"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Di", "cell", dim)
        axis = Variable("material__Di_axis", "cell", dim, (3,))
        return D * (m.dot(gradient(m.dot(axis))) - divergence(m) * m.dot(axis))
