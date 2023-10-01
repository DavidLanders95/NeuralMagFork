from ..generators import pytorch_generator as gen
from ..common import Function, VectorFunction, logging, config
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
            if config.torch['compile']:
                self.h_func = torch.compile(self._code.h)
            else:
                self.h_func = self._code.h
        logging.info_green(f"[{self.__class__.__name__}] Register state methods (field: '{h_name or self._h_name}')")
        setattr(state, h_name or self._h_name, (self.h_func, 'node', (3,)))

    @classmethod
    def generate_code(cls):
        code = ''

        # generate linear-form code
        m = gen.Variable('m', 'node', (3,))
        field_expr = gen.gateaux_derivative(cls.e_expr(m), m)
        cmds1, vars1 = gen.linear_form_cmds(field_expr)

        # generate lumped mass code
        v = gen.Variable('v', 'node')
        Ms = gen.Variable('material__Ms', 'cell')
        cmds2, vars2 = gen.linear_form_cmds(- constants.mu_0 * Ms * v)

        code = gen.CodeBlock()
        with code.add_function('h', sorted(list(vars1 | vars2 | {'m'}))) as f:
            f.zeros_like('h', 'm')
            for cmd in cmds1:
                f.add_to('h', cmd[0], cmd[1])

            f.zeros_like('mass', 'm', shape = 'm.shape[:3]')
            for cmd in cmds2:
                f.add_to('mass', cmd[0], cmd[1])

            f.retrn('h / mass.unsqueeze(-1)')

        return code
