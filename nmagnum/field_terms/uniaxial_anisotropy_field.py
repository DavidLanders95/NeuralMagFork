from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N

__all__ = ['UniaxialAnisotropyField']

class UniaxialAnisotropyField(FieldTerm):
    _h_name = 'h_uaniso'

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m):
        K = Variable('material.Ku', 'dg')
        axis = Variable('material.Ku_axis', 'dg', (3,))
        return - K * m.dot(axis)**2
