from ..generators import pytorch_generator as gen
from ..common import Function, VectorFunction
from scipy import constants
import sys
import inspect
import torch

__all__ = ["FieldTerm"]

class Code(): pass

class FieldTerm(gen.CodeClass):
    def __init__(self, state, *args, **kwargs):
        self.code = Code()
        if hasattr(self, 'e_expr'):
            super().__init__(generate_code = True, **kwargs)

            # set up scratch space for h
            self._h = VectorFunction(state)
            self._args = [self._h.tensor, state.mesh.dx]
            args = list(inspect.signature(self.code.h).parameters.keys())
            for arg in args[2:]:
                container = state
                while '__' in arg:
                    parent, child = arg.split('__', 1)
                    container = getattr(container, parent)
                    arg = child
                self._args.append(getattr(container, arg).tensor)
        else:
            super().__init__(generate_code = False, **kwargs)
            self.code = Code()
            self._args = []

    def h(self):
        self.code.h(*self._args)
        return self._h

    @classmethod
    def generate_code(cls):
        code = ''

        # generate linear-form code
        m = gen.Variable('m', 'cg', (3,))
        field_expr = gen.gateaux_derivative(cls.e_expr(m), m)
        cmds, variables = gen.linear_form_cmds(field_expr, 'h')
        variables.add('material__Ms')

        # write header
        code += "import torch\n"
        code += "@torch.compile\n"
        code += f"def h(h, dx, {', '.join(sorted(variables))}):\n"

        # linear form
        code += "    h[:] = 0\n"
        for lhs, rhs in cmds.items():
            code += f"    {lhs} += {rhs}\n"

        # inverse mass
        v = gen.Variable('v', 'cg')
        Ms = gen.Variable('material__Ms', 'dg')
        cmds, _ = gen.linear_form_cmds(- constants.mu_0 * Ms * v, 'mass')
        code += "    mass = torch.zeros(h.shape[:3], dtype = h.dtype, device = h.device)\n"
        for lhs, rhs in cmds.items():
            code += f"    {lhs} += {rhs}\n"
        code += "    h /= mass.unsqueeze(-1)\n"

        return code
