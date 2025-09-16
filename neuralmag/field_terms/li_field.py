from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["LIField"]


class LIField(FieldTerm):
    r"""
    TODO
    """
    default_name = "LI"

    def __init__(self, LI: str, **kwargs):
        super().__init__(**kwargs)
        self.default_name = f"LI_{LI}"
        self._options = LI
        self._validate_LI()

    def _validate_LI(self):
        """Validate and convert LI and ensure it is a valid string of x,y,z."""
        valid_components = ["x", "y", "z"]

        if not isinstance(self._options, str):
            raise TypeError(
                f"LI must be a string, but got {self._options} of type {type(self._options).__name__}."
            )

        if len(self._options) != 3:
            raise ValueError(
                f"LI must be a string of length 3, but got {self._options} of length {len(self._options)}."
            )

        for char in self._options:
            if char not in valid_components:
                raise ValueError(
                    f"LI contains invalid character '{char}'. Valid characters are 'x', 'y', and 'z'."
                )

    @staticmethod
    def e_expr(m, dim, options):
        D = Variable(f"material__D{options}", "c" * dim)
        return D * Lifshitz_invariant(m, options) * dV(dim)


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
