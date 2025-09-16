"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy
import threading
from collections.abc import Iterable

import sympy as sp

from neuralmag.common.engine import N, Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["DMIField"]


class DMIField(FieldTerm):
    r"""
    Effective field contribution for the micromagnetic bulk-DMI energy

    .. math::

      E = \int_\Omega D \vec{m} \cdot (\nabla \times \vec{m}) \dx

    with the DMI constant :math:`D` given in units of :math:`\text{J/m}^2`.

    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Db** (*cell scalar field*) The DMI constant in J/m^2
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "dmi"

    _thread_local = threading.local()

    def __init__(self, LI=None, **kwargs):
        super().__init__(**kwargs)
        if LI is None:
            raise ValueError("LI must be provided as an argument.")
        self.LI = copy.deepcopy(LI)
        self._validate_LI()
        self._print_LI_form(LI)
        # Set LI for the current thread
        DMIField._thread_local.LI = self.LI

    def _validate_LI(self):
        """Validate and convert LI, ensuring it uses valid components."""
        valid_components = {"x": N.x, "y": N.y, "z": N.z}

        if not isinstance(self.LI, Iterable):
            raise TypeError(
                f"LI must be an iterable (e.g., list, tuple), but got {self.LI} of type {type(self.LI).__name__}."
            )

        for index, sublist in enumerate(self.LI):
            if not isinstance(sublist, Iterable):
                raise TypeError(
                    f"Element at index {index} of LI must be an iterable (e.g., list, tuple), "
                    f"but got {sublist} of type {type(sublist).__name__}."
                )
            for inner_index, inner_list in enumerate(sublist):
                if not isinstance(inner_list, Iterable):
                    raise TypeError(
                        f"Element at index {inner_index} of sublist {index} in LI must be an iterable "
                        f"(e.g., list, tuple), but got {inner_list} of type {type(inner_list).__name__}."
                    )
                if len(inner_list) != 3:
                    raise ValueError(
                        f"Inner iterable at index {inner_index} of sublist {index} in LI must have exactly 3 elements, "
                        f"but got {inner_list} with {len(inner_list)} elements."
                    )
                # Convert strings to valid components
                for i, component in enumerate(inner_list):
                    if component not in valid_components:
                        raise ValueError(
                            f"Invalid component '{component}' in inner iterable at index {inner_index} of sublist {index}. "
                            f"Expected one of {list(valid_components.keys())}."
                        )
                    # Replace string with corresponding N.x, N.y, or N.z
                    inner_list[i] = valid_components[component]

    @staticmethod
    def _print_LI_form(LI):
        """Print the LI in the form of a long sum with D coefficients."""
        # Initialize LaTeX printing if available
        sp.init_printing(use_latex="mathjax", pretty_print=True)

        # Build the symbolic expression
        terms = []
        for i, sublist in enumerate(LI):
            D_symbol = sp.symbols(f"D_{i}")
            temp = []
            for inner_list in sublist:
                i_comp, j_comp, k_comp = inner_list
                # Create symbolic representations for D_i and \mathcal{L}
                L_symbol = sp.Symbol(
                    f"\\mathcal{{L}}_{{{i_comp}{j_comp}}}^{{{k_comp}}}"
                )
                temp.append(L_symbol)
            terms.append(D_symbol * sp.Add(*temp))

        # Combine all terms into a symbolic expression
        expression = sp.Add(*terms)

        # Print the expression in LaTeX format if supported
        try:
            from IPython.display import display

            display(expression)
        except ImportError:
            # Fallback to pretty printing if LaTeX rendering is unavailable
            sp.pprint(expression)

    @staticmethod
    def e_expr(m, dim):
        LI = getattr(DMIField._thread_local, "LI", None)
        if LI is None:
            raise RuntimeError(
                "LI is not set. Ensure DMIField is initialized properly."
            )
        l_LI = len(LI)
        ed = 0
        for i in range(l_LI):
            D = Variable(f"material__D{i}", "c" * dim)
            l2 = len(LI[i])
            ed += D * sum(Lifshitz_invariant(m, *LI[i][j]) for j in range(l2))
        return ed * dV()


def Lifshitz_invariant(m, i, j, k):
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
