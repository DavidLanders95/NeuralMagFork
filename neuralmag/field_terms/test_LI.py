import threading

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["LIField"]


class LIField(FieldTerm):
    r"""
    TODO
    """

    _thread_local = threading.local()

    def __init__(self, LI: str, **kwargs):
        super().__init__(**kwargs)
        self.LI = LI
        self._validate_LI()
        LIField._thread_local.LI = self.LI
        self.default_name = f"dmi_{self.LI}"

    def _validate_LI(self):
        """Validate and convert LI and ensure it is a valid string of x,y,z."""
        valid_components = ["x", "y", "z"]

        if not isinstance(self.LI, str):
            raise TypeError(
                f"LI must be a string, but got {self.LI} of type {type(self.LI).__name__}."
            )

        if len(self.LI) != 3:
            raise ValueError(
                f"LI must be a string of length 3, but got {self.LI} of length {len(self.LI)}."
            )

        for char in self.LI:
            if char not in valid_components:
                raise ValueError(
                    f"LI contains invalid character '{char}'. Valid characters are 'x', 'y', and 'z'."
                )

    @staticmethod
    def e_expr(m, dim):
        LI = getattr(LIField._thread_local, "LI", None)
        D = Variable(f"material__D{LI}", "c" * dim)
        return D * Lifshitz_invariant(m, LI) * dV()


def Lifshitz_invariant(m, LI):
    valid_components = {"x": N.x, "y": N.y, "z": N.z}
    i, j, k = [valid_components[l] for l in LI]
    i_s = component(i)
    j_s = component(j)
    return m.dot(i_s) * m.dot(j_s).diff(k) - m.dot(j_s) * m.dot(i_s).diff(k)


def component(a):
    match a:
        case N.x:
            return N.i
        case N.y:
            return N.j
        case N.z:
            return N.k
        case _:
            raise ValueError(f"{a} is not N.x, N.y, or N.z.")
