# SPDX-License-Identifier: MIT

import inspect
import types

from scipy import constants

from neuralmag.common import VectorFunction, config
from neuralmag.common.engine import Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["ExternalField"]


class ExternalField(FieldTerm):
    r"""
    Effective field contribution corresponding to the external field

    .. math::

      E = - \int_\Omega \mu_0 M_s  \vec{m} \cdot \vec{h} \dx

    :param h: The field either given as a :code:`config.backend.Tensor` or callable
              in case of a field that depends e.g. on :code:`state.t`.
              The shape must be either (nx, ny, nz, 3) or (3,) in which case
              the field expanded to full size.
    :type h: config.backend.Tensor
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
            # check if h function return 3-vector
            value = state.resolve(self._h, [])()
            if value.shape == (3,):
                arg_names = list(inspect.signature(self._h).parameters.keys())
                block = config.backend.CodeBlock(plain=True)
                with block.add_function("h", arg_names) as func:
                    func.retrn_expanded(f"__h({', '.join(arg_names)})", tensor_shape)
                compiled_code = compile(str(block), "<string>", "exec")
                self.h = types.FunctionType(
                    compiled_code.co_consts[0],
                    {**config.backend.libs, **{f"__h": self._h}},
                    name,
                )
            else:
                self.h = self._h
        elif isinstance(self._h, config.backend.Tensor):
            if self._h.shape == tensor_shape:
                self.h = self._h
            elif self._h.shape == (3,):
                self.h = config.backend.broadcast_to(self._h, tensor_shape)
            else:
                raise Exception("Shape not matching")
        elif isinstance(self._h, VectorFunction):
            self.h = self._h.tensor
        else:
            raise Exception("Type not supported")

        super().register(state, name)
        # fix reference to h_external in E_external if suffix is changed
        if name is not None:
            func = state.remap(self.E, {"h_external": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), func)

    @staticmethod
    def e_expr(m, dim):
        Ms = Variable("material__Ms", "c" * dim)
        h_external = Variable("h_external", "n" * dim, (3,))
        return -constants.mu_0 * Ms * m.dot(h_external) * dV(dim)
