from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["UniaxialAnisotropyField"]


class UniaxialAnisotropyField(FieldTerm):
    _name = "uaniso"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        K = Variable("material__Ku", "c" * dim)
        axis = Variable("material__Ku_axis", "c" * dim, (3,))
        return -K * m.dot(axis) ** 2 * dV(dim)
