from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["ExchangeField"]


class ExchangeField(FieldTerm):
    _name = "exchange"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        A = Variable("material__A", "c" * dim)
        return (
            A
            * (
                m.diff(N.x).dot(m.diff(N.x))
                + m.diff(N.y).dot(m.diff(N.y))
                + m.diff(N.z).dot(m.diff(N.z))
            )
            * dV(dim)
        )
