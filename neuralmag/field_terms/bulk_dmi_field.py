from sympy.vector import curl

from ..generators.pytorch_generator import N, Variable
from .field_term import FieldTerm

__all__ = ["BulkDMIField"]


class BulkDMIField(FieldTerm):
    _name = "bdmi"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Db", "cell", dim)
        return D * m.dot(curl(m))
