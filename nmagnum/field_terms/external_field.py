from .field_term import FieldTerm
from ..generators.pytorch_generator import Variable, N
from scipy import constants

__all__ = ['ExternalField']

class ExternalField(FieldTerm):
    _name = 'external'
    h = None

    def __init__(self, h, *args, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    @staticmethod
    def e_expr(m):
        Ms = Variable('material__Ms', 'cell')
        h_external = Variable('h_external', 'node', (3,))
        return - constants.mu_0 * Ms * m.dot(h_external)
