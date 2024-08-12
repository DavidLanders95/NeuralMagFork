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

import torch

from ..generators.pytorch_generator import N, Variable, dA
from .field_term import FieldTerm

__all__ = ["InterlayerExchangeField"]


def swap(m, iidx):
    result = torch.zeros_like(m)
    result[:, :, iidx[0], :] = m[:, :, iidx[1], :]
    result[:, :, iidx[1], :] = m[:, :, iidx[0], :]
    return result


class InterlayerExchangeField(FieldTerm):
    r"""
    Effective field contribution for interface exchange up to bbiquadratic order

    .. math::

      E = - 0.5 \int_\Gamma A \vec{m} \cdot \vec{m}_\text{other} \ds

    where :math:`\Gamma` denotes two coupled interfaces and
    :math:`\vec{m}_\text{other}` denotes the magnetization at the nearest
    point of the other interface. This expression is equivalent to the more
    common expression

    .. math::

      E = - \int_\Gamma A \vec{m}_1 \cdot \vec{m}_2 \ds

    where :math:`\Gamma` denotes a shared interface and :math:`\vec{m}_1`
    and :math:`\vec{m}_2` denote the magnetization on the respective sides.
    Currently, only surfaces in the xy-plane are supported.

    :param idx1: z-index of the first interfaces
    :type idx1: int
    :param idx2: z-index of the second interfaces
    :type idx2: int
    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.iA** (*CCN scalar field*) Interface coupling constant in J/m^2
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "iexchange"

    def __init__(self, idx1, idx2, **kwargs):
        super().__init__(**kwargs)
        self._iidx = [idx1, idx2]

    def register(self, state, name=None):
        super().register(state, name)

        state.iidx = torch.tensor(self._iidx, device=state.device, dtype=torch.int)
        state.im_other = (swap, "node", (3,))

    @staticmethod
    def e_expr(m, dim):
        assert dim == 3
        iA = Variable("material__iA", "ccn")
        im_other = Variable("im_other", "nnn", (3,))

        return -0.5 * iA * m.dot(im_other) * dA(dim, idx="iidx")

    @staticmethod
    def dedm_expr(m, dim):
        assert dim == 3
        v = Variable("v", "nnn", (3,))
        iA = Variable("material__iA", "ccn")
        im_other = Variable("im_other", "nnn", (3,))

        return -iA * v.dot(im_other) * dA(dim, idx="iidx")
