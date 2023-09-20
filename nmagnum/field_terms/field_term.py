from ..generators import pytorch_generator as gen
from ..common import Function, VectorFunction, logging
from scipy import constants
import sys
import inspect
import torch

__all__ = ["FieldTerm"]

class FieldTerm(gen.CodeClass):

    def __init__(self, **kwargs):
        pass

    def register(self, state, h_name = None):
        super().__init__(generate_code = hasattr(self, 'e_expr'))
        if not hasattr(self, 'h_func'):
            try:
                logging.info_green(f"[{self.__class__.__name__}] Compile field method")
                self.h_func = torch.compile(self._code.h)
            except:
                self.h_func = self._code.h
        logging.info_green(f"[{self.__class__.__name__}] Register state methods (field: '{h_name or self._h_name}')")
        setattr(state, h_name or self._h_name, (self.h_func, 'node', (3,)))

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
        #code += "@torch.compile\n"
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
