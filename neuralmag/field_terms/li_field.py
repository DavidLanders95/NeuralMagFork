from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["LIField"]


class LIField(FieldTerm):
    r"""
    Effective field contribution for a single Lifshitz invariant (LI) term.

    These terms can be combined to form the DMI energy for different crystal
    symmetries. Lifshitz invariants are of the form

    .. math::

        L_{ij}^k(\vec m) = m_i \partial_k m_j - m_j \partial_k m_i,

    where :math:`i,j,k \in \{x,y,z\}`, :math:`i \neq j`, and :math:`m_i` is the
    :math:`i`-th Cartesian component of the unit magnetization vector
    :math:`\vec m`. This class constructs the energy density by multiplying
    the Lifshitz invariant by a DMI constant, giving the form

    .. math::

        E = - \int_\Omega D_{ijk} L_{ij}^k(\vec m) \dx,

    with material constant :math:`D_{ijk}` provided as a scalar cell field whose
    name encodes the chosen LI directions.

    The specific invariant is selected by the string ``LI`` passed to the
    constructor. This must be a three-character string composed only of the
    letters ``x``, ``y`` and ``z``. The first two characters specify the pair
    :math:`(i,j)` in order whilst the third character specifies
    :math:`k` (the derivative direction).

    The corresponding material constant must be supplied in the simulation state
    as ``state.material.D{LI}`` (e.g. ``state.material.Dxyz``).

    :param LI: Three-character string selecting the Lifshitz invariant indices (e.g. ``'xyz'``).
    :type LI: str
    :param n_gauss: Degree of Gauss quadrature used in the form compiler (inherited via kwargs).
    :type n_gauss: int, optional

    :Required state attributes (if not renamed):
         * **state.material.D{LI}** (*cell scalar field*) The DMI constant associated with this invariant.
         * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m.

    :raises TypeError: If ``LI`` is not a string.
    :raises ValueError: If ``LI`` length is not 3 or contains characters other than ``x``, ``y``, ``z``.

    :Example:
    .. code-block::

        state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

        state.material.Dxyz = 1e-3
        nm.LIField("xyz").register(state, "LI_xyz")

    """

    default_name = "LI"

    def __init__(self, LI: str, **kwargs):
        super().__init__(**kwargs)
        self.default_name = f"LI_{LI}"
        self._options = {"LI": LI}
        self._validate_LI()

    def _validate_LI(self):
        """Validate and convert LI and ensure it is a valid string of x,y,z."""
        valid_components = ["x", "y", "z"]

        if not isinstance(self._options["LI"], str):
            raise TypeError(
                f"LI must be a string, but got {self._options['LI']} of type {type(self._options['LI']).__name__}."
            )

        if len(self._options["LI"]) != 3:
            raise ValueError(
                f"LI must be a string of length 3, but got {self._options['LI']} of length {len(self._options['LI'])}."
            )

        for char in self._options["LI"]:
            if char not in valid_components:
                raise ValueError(
                    f"LI contains invalid character '{char}'. Valid characters are 'x', 'y', and 'z'."
                )

    @staticmethod
    def e_expr(m, dim, options):
        LI = options["LI"]
        D = Variable(f"material__D{LI}", "c" * dim)
        return D * Lifshitz_invariant(m, LI) * dV(dim)


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
