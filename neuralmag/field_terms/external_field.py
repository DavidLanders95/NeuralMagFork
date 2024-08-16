"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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

    :param h: The field either given as a :code:`torch.Tensor` or callable
              in case of a field that depends e.g. on :code:`state.t`.
              The shape must be either (nx, ny, nz, 3) or (3,) in which case
              the field expanded to full size.
    :type h: torch.Tensor
    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m

    :Example:
        .. code-block::

            state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))

            # define constant external field from expanded function
            h_ext = nm.VectorFunction(state).fill((0, 0, 0), expand=True)
            external = nm.ExternalField(h_ext)

            # define external field in y-direction linearly increasing with time
            external = nm.ExternalField(lambda t: t * state.tensor([0, 8e5 / 10e-9, 0]))

    """
    default_name = "external"
    h = None

    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self._h = h

    def register(self, state, name=None):
        tensor_shape = VectorFunction(state).tensor_shape
        if callable(self._h):
            func, args = state.get_func(self._h)
            value = func(*args)
            if value.shape == (3,):
                arg_names = list(inspect.signature(func).parameters.keys())
                code = f"def h({', '.join(arg_names)}):\n"
                code += (
                    f"    return __h({', '.join(arg_names)}).expand({tensor_shape})\n"
                )
                compiled_code = compile(code, "<string>", "exec")
                self.h = types.FunctionType(
                    compiled_code.co_consts[0], {f"__h": self._h}, name
                )
            else:
                self.h = self._h
        elif isinstance(self._h, torch.Tensor):
            if self._h.shape == tensor_shape:
                self.h = self._h
            elif self._h.shape == (3,):
                self.h = self._h.expand(tensor_shape)
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
