from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N

__all__ = ["ExchangeField"]

class ExchangeField(FieldTerm):
    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

    @staticmethod
    def e_expr(m):
        A = Variable('material.A', 'dg')
        return A * (
                m.diff(N.x).dot(m.diff(N.x)) + 
                m.diff(N.y).dot(m.diff(N.y)) +
                m.diff(N.z).dot(m.diff(N.z))
                )
