from ..generators import pytorch_generator as gen
from ..common import Function, VectorFunction
from scipy import constants
import sys
import inspect
import torch

__all__ = ["FieldTerm"]

class FieldTerm(gen.CodeClass):

    def __init__(self, state, **kwargs):
        super().__init__(generate_code = hasattr(self, 'e_expr'), **kwargs)
        self._state = state

        if not hasattr(self, 'h_func'):
            self.h_func = self._code.h

    def h(self):
        if not hasattr(self, '_h_args'):
            self._h_args = self.args_for_function(self.h_func)
        return VectorFunction(self._state, tensor = self.h_func(*self._h_args))

    # TODO create args_for_h?
    def args_for_function(self, f):
        result = []
        for arg in list(inspect.signature(f).parameters.keys()):
            container = self._state
            while '__' in arg:
                parent, child = arg.split('__', 1)
                container = getattr(container, parent)
                arg = child
            attr = getattr(container, arg)
            if hasattr(attr, 'tensor'):
                result.append(attr.tensor)
            else:
                result.append(attr)
        return result

    @classmethod
    def generate_code(cls):
        code = ''

        # generate linear-form code
        m = gen.Variable('m', 'cg', (3,))
        field_expr = gen.gateaux_derivative(cls.e_expr(m), m)
        cmds, variables = gen.linear_form_cmds(field_expr, 'h')
        variables.add('m')
        variables.add('material__Ms')

        # write header
        code += "import torch\n"
        code += "@torch.compile\n"
        code += f"def h(mesh__dx, {', '.join(sorted(variables))}):\n"
        code += "    dx = mesh__dx\n"
        code += "    h = torch.zeros(m.shape, dtype = m.dtype, device = m.device)\n"

        # linear form
        for lhs, rhs in cmds.items():
            code += f"    {lhs} += {rhs}\n"

        # inverse mass
        v = gen.Variable('v', 'cg')
        Ms = gen.Variable('material__Ms', 'dg')
        cmds, _ = gen.linear_form_cmds(- constants.mu_0 * Ms * v, 'mass')
        code += "    mass = torch.zeros(m.shape[:3], dtype = m.dtype, device = m.device)\n"
        for lhs, rhs in cmds.items():
            code += f"    {lhs} += {rhs}\n"
        code += "    h /= mass.unsqueeze(-1)\n"
        code += "    return h\n"

        return code
