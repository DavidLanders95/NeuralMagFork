import inspect
import types

import torch
from scipy import constants

from ..common import VectorFunction
from ..generators.pytorch_generator import Variable, dV
from .field_term import FieldTerm

__all__ = ["ExternalField"]


class ExternalField(FieldTerm):
    r"""
    Effective field contribution corresponding to the external field

    .. math::

      E = - \int_\Omega \mu_0 M_s  \vec{m} \cdot \vec{h} \dx
    """
    _name = "external"
    h = None

    def __init__(self, h, expand=False, **kwargs):
        super().__init__(**kwargs)
        self._h = h

    def register(self, state, name=None):
        size = VectorFunction(state).size
        if callable(self._h):
            func, args = state.get_func(self._h)
            value = func(*args)
            if value.shape == (3,):
                arg_names = list(inspect.signature(func).parameters.keys())
                code = f"def h({', '.join(arg_names)}):\n"
                code += f"    return __h({', '.join(arg_names)}).expand({size})\n"
                compiled_code = compile(code, "<string>", "exec")
                self.h = types.FunctionType(
                    compiled_code.co_consts[0], {f"__h": self._h}, name
                )
            else:
                self.h = self._h
        elif isinstance(self._h, torch.Tensor):
            if self._h.shape == size:
                self.h = self._h
            elif self._h.shape == (3,):
                self.h = self._h.expand(size)
            else:
                raise Exception("Shape not matching")
        elif isinstance(self._h, VectorFunction):
            self.h = self._h.tensor
        else:
            raise Exception("Type not supported")

        super().register(state, name)
        # fix reference to h_external in E_external if suffix is changed
        if name is not None:
            wrapped = state.wrap_func(self.E, {"h_external": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), wrapped)

    @staticmethod
    def e_expr(m, dim):
        Ms = Variable("material__Ms", "c" * dim)
        h_external = Variable("h_external", "n" * dim, (3,))
        return -constants.mu_0 * Ms * m.dot(h_external) * dV(dim)
