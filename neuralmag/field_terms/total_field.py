# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import types

from neuralmag.common import logging
from neuralmag.field_terms.field_term import FieldTerm

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
