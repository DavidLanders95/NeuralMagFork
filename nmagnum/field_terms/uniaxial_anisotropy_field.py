from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N

__all__ = ['UniaxialAnisotropyField']

class UniaxialAnisotropyField(FieldTerm):
    _name = 'uaniso'

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m):
        K = Variable('material__Ku', 'cell')
        axis = Variable('material__Ku_axis', 'cell', (3,))
        return - K * m.dot(axis)**2
