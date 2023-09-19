from .field_term import FieldTerm
import types

__all__ = ['TotalField']

class TotalField(FieldTerm):
    _h_name = 'h_total'

    def __init__(self, *field_names):
        self._field_names = field_names

    def register(self, state, h_name = None):
        code  = f"def h_total({', '.join(self._field_names)}):\n"
        code += f"    return {' + '.join(self._field_names)}"
        compiled_code = compile(code, "<string>", "exec")
        func = types.FunctionType(compiled_code.co_consts[0], {}, 'h_total')
        setattr(state, h_name or self._h_name, (func, 'node', (3,)))
