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
    def __init__(self, *args, **kwargs):
        # initialize code-file cache
        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)
        code_dir = this_module.parent / 'code'
        code_dir.mkdir(parents = True, exist_ok = True)
        code_module_name = f"{this_module.stem}_code"
        self._code_file_path = code_dir / f"{code_module_name}.py"

        # generate code
        if not self._code_file_path.is_file():
            self.generate_code()

        # import code
        spec = importlib.util.spec_from_file_location(code_module_name, self._code_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[code_module_name] = module
        spec.loader.exec_module(module)
        self._code = module

    def h(self, state):
        if not hasattr(state, '_args'):
            # set up scratch space for h
            self._h = VectorFunction(state)
            self._args = [self._h.tensor, state.mesh.dx]
            args = list(inspect.signature(self._code.compute_heff).parameters.keys())
            for arg in args[2:]:
                container = state
                while '__' in arg:
                    parent, child = arg.split('__', 1)
                    container = getattr(container, parent)
                    arg = child
                self._args.append(getattr(container, arg).tensor)

        self._code.compute_heff(*self._args)

        return self._h

    @property
    def hraw(self):
        return self._code.compute_heff

    def generate_code(self):
        with open(self._code_file_path, 'w') as code_file:
            # generate linear-form code
            m = gen.Variable('m', 'cg', (3,))
            field_expr = gen.gateaux_derivative(self.e_expr(m), m)
            cmds, variables = gen.linear_form_cmds(field_expr, 'h')
            variables.add('material__Ms')

            # write header
            code_file.write("import torch\n")
            code_file.write("@torch.compile\n")
            code_file.write(f"def compute_heff(h, dx, {', '.join(sorted(variables))}):\n")

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
