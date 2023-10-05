from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N
from sympy.vector import gradient, divergence

__all__ = ['InterfaceDMIField']

class InterfaceDMIField(FieldTerm):
    _name = 'idmi'

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable('material__Di', 'cell', (), dim)
        axis = Variable('material__Di_axis', 'cell', (3,), dim)
        return D * (m.dot(gradient(m.dot(axis))) - divergence(m) * m.dot(axis))
