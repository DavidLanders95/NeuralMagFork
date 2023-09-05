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
        this_module = pathlib.Path(importlib.import_module(self.__module__).__file__)

        code_dir = this_module.parent / 'code'
        code_dir.mkdir(parents = True, exist_ok = True)

        code_module_name = f"{this_module.stem}_code"
        code_file_path = code_dir / f"{code_module_name}.py"

        if not code_file_path.is_file():

            # generate the code
            with open(code_file_path, 'w') as code_file:
                m = gen.Variable('m', 'cg', (3,))
                e_expr = self.e_expr(m)
                field_expr = gen.gateaux_derivative(e_expr, m)

                code_file.write("import torch\n")
                code_file.write(gen.assemble_linear_form(field_expr))

        # import code
        spec = importlib.util.spec_from_file_location(code_module_name, code_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[code_module_name] = module
        spec.loader.exec_module(module)
        self._code = module

    def h(self, state):
        if not hasattr(state, '_args'):
            # set up scratch space for h
            self._h = VectorFunction(state)

            self._args = [self._h.tensor, state.mesh.dx]
            args = list(inspect.signature(self._code.assemble_linear_form).parameters.keys())
            for arg in args[2:]:
                container = state
                while '__' in arg:
                    parent, child = arg.split('__', 1)
                    container = getattr(container, parent)
                    arg = child
                self._args.append(getattr(container, arg).tensor)

            print(self._args)

        self._code.assemble_linear_form(*self._args)

        # compute lumped mass
        V = state.mesh.cell_volume
        Ms = state.material.Ms.tensor
        mass = Function(state).tensor
        mass[:-1, :-1, :-1] += V / 8.0 * Ms
        mass[:-1, :-1, 1:] += V / 8.0 * Ms
        mass[:-1, 1:, :-1] += V / 8.0 * Ms
        mass[:-1, 1:, 1:] += V / 8.0 * Ms
        mass[1:, :-1, :-1] += V / 8.0 * Ms
        mass[1:, :-1, 1:] += V / 8.0 * Ms
        mass[1:, 1:, :-1] += V / 8.0 * Ms
        mass[1:, 1:, 1:] += V / 8.0 * Ms

        self._h.tensor.multiply_(-1.0 / (constants.mu_0 * mass.unsqueeze(-1)))

        return self._h
