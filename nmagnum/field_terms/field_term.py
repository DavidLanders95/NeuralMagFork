from ..generators import pytorch_generator as gen
from ..common import Function, VectorFunction
from scipy import constants
import pathlib
import importlib
import sys
import inspect
import torch

__all__ = ["FieldTerm"]

class FieldTerm(object):
    def __init__(self, state, *args, **kwargs):
        # generate code
        if not self._code_file_path().is_file():
            self.generate_code()

        # import code
        module_spec = importlib.util.spec_from_file_location('code', self._code_file_path())
        self.code = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.code)

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

    def h(self):
        self.code.h(*self._args)
        return self._h

    @classmethod
    def _code_file_path(cls):
        this_module = pathlib.Path(importlib.import_module(cls.__module__).__file__)
        return this_module.parent / 'code' / f"{this_module.stem}_code.py"

    @classmethod
    def generate_code(cls):
        path = cls._code_file_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, 'w') as code_file:
            # generate linear-form code
            m = gen.Variable('m', 'cg', (3,))
            field_expr = gen.gateaux_derivative(cls.e_expr(m), m)
            cmds, variables = gen.linear_form_cmds(field_expr, 'h')
            variables.add('material__Ms')

            # write header
            code_file.write("import torch\n")
            code_file.write("@torch.compile\n")
            code_file.write(f"def h(h, dx, {', '.join(sorted(variables))}):\n")

            # linear form
            code_file.write("    h[:] = 0\n")
            for lhs, rhs in cmds.items():
                code_file.write(f"    {lhs} += {rhs}\n")

            # inverse mass
            v = gen.Variable('v', 'cg')
            Ms = gen.Variable('material__Ms', 'dg')
            cmds, _ = gen.linear_form_cmds(- constants.mu_0 * Ms * v, 'mass')
            code_file.write("    mass = torch.zeros(h.shape[:3], dtype = h.dtype, device = h.device)\n")
            for lhs, rhs in cmds.items():
                code_file.write(f"    {lhs} += {rhs}\n")
            code_file.write("    h /= mass.unsqueeze(-1)\n")
