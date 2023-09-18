from .field_term import FieldTerm
import types

__all__ = ['TotalField']

class TotalField(FieldTerm):
    def __init__(self, *field_names):
        self._field_names = field_names
        self.h_name = 'h_total'

    def register(self, state):
        code  = f"def h({', '.join(self._field_names)}):\n"
        code += f"    return {' + '.join(self._field_names)}"
        compiled_code = compile(code, "<string>", "exec")
        func = types.FunctionType(compiled_code.co_consts[0], globals(), 'h')
        setattr(state, self.h_name, (func, 'node', (3,)))
