from scipy import constants

from ..generators.pytorch_generator import Variable, dV
from .field_term import FieldTerm

__all__ = ["ExternalField"]


class ExternalField(FieldTerm):
    _name = "external"
    h = None

    def __init__(self, h, *args, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def register(self, state, name=None):
        super().register(state, name)
        # fix reference to h_external in E_external if suffix is changed
        if name is not None:
            wrapped = state.wrap_func(self.E, {"h_external": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), wrapped)

    @staticmethod
    def e_expr(m, dim):
        Ms = Variable("material__Ms", "cell", dim)
        h_external = Variable("h_external", "node", dim, (3,))
        return -constants.mu_0 * Ms * m.dot(h_external) * dV()
