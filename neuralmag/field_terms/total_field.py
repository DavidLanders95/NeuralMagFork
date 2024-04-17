import types

from ..common import logging
from .field_term import FieldTerm

__all__ = ["TotalField"]


class TotalField(FieldTerm):
    _name = ""

    def __init__(self, *field_names):
        self._field_names = field_names

    def register(self, state, name=None):
        code = (
            f"def h_total({', '.join([self.attr_name('h', name) for name in self._field_names])}):\n"
        )
        code += (
            "    return"
            f" {' + '.join([self.attr_name('h', name) for name in self._field_names])}"
        )
        compiled_code = compile(code, "<string>", "exec")
        h_func = types.FunctionType(compiled_code.co_consts[0], {}, "h_total")

        code = (
            f"def E_total({', '.join([self.attr_name('E', name) for name in self._field_names])}):\n"
        )
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
