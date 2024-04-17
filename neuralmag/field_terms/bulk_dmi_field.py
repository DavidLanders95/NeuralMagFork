from sympy.vector import curl

from ..generators.pytorch_generator import N, Variable, dV
from .field_term import FieldTerm

__all__ = ["BulkDMIField"]


class BulkDMIField(FieldTerm):
    r"""
    Effective field contribution for the micromagnetic bulk-DMI energy

    .. math::

      E = \int_\Omega D \vec{m} \cdot (\nabla \times \vec{m}) \dx

    with the DMI constant :math:`D` given in units of :math:`\text{J/m}^2`.
    """
    _name = "bdmi"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def e_expr(m, dim):
        D = Variable("material__Db", "c" * dim)
        return D * m.dot(curl(m)) * dV()
