# SPDX-License-Identifier: MIT

import types

from neuralmag.common import VectorFunction, logging
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
        setattr(state, self.attr_name("h", name), VectorFunction(state, tensor=h_func))
        setattr(state, self.attr_name("E", name), E_func)
