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

import types

from ..common import logging
from .field_term import FieldTerm

__all__ = ["TotalField"]


class TotalField(FieldTerm):
    r"""
    This class combines multiple field terms into a single total field term by
    adding up their effective fields and energies.

    :param \*field_names: The names of the effective field contributions

    :Example:
        .. code-block::

            state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

            nm.ExchangeField().register(state, "exchange")
            nm.DemagField().register(state, "demag")
            nm.ExternalField(h_ext).register(state, "external")
            nm.TotalField("exchange", "demag", "external").register(state)

            # Compute total field and energy
            h = state.h
            E = state.E
    """
    default_name = ""

    def __init__(self, *field_names):
        self._field_names = field_names

    def register(self, state, name=None):
        code = f"def h_total({', '.join([self.attr_name('h', name) for name in self._field_names])}):\n"
        code += (
            "    return"
            f" {' + '.join([self.attr_name('h', name) for name in self._field_names])}"
        )
        compiled_code = compile(code, "<string>", "exec")
        h_func = types.FunctionType(compiled_code.co_consts[0], {}, "h_total")

        code = f"def E_total({', '.join([self.attr_name('E', name) for name in self._field_names])}):\n"
        code += (
            "    return"
            f" {' + '.join([self.attr_name('E', name) for name in self._field_names])}"
        )
        compiled_code = compile(code, "<string>", "exec")
        E_func = types.FunctionType(compiled_code.co_consts[0], {}, "E_total")

        logging.info_green(
            f"[{self.__class__.__name__}] Register state methods (field:"
            f" '{self.attr_name('h', name)}', energy: '{self.attr_name('E', name)}')"
        )
        setattr(state, self.attr_name("h", name), (h_func, "n" * state.mesh.dim, (3,)))
        setattr(state, self.attr_name("E", name), E_func)
