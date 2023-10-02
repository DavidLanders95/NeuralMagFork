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

    def register(self, state, name = None):
        super().register(state, name)
        # fix reference to h_external in E_external if suffix is changed
        if name is not None:
            wrapped = state.wrap_func(self.E, {'h_external': self.attr_name('h', name)})
            setattr(state, self.attr_name('E', name), wrapped)

    @staticmethod
    def e_expr(m):
        Ms = Variable('material__Ms', 'cell')
        h_external = Variable('h_external', 'node', (3,))
        return - constants.mu_0 * Ms * m.dot(h_external)
