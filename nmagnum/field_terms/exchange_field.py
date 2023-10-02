from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N

__all__ = ['ExchangeField']

class ExchangeField(FieldTerm):
    _name = 'exchange'

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m):
        A = Variable('material__A', 'cell')
        return A * (
                m.diff(N.x).dot(m.diff(N.x)) + 
                m.diff(N.y).dot(m.diff(N.y)) +
                m.diff(N.z).dot(m.diff(N.z))
                )
