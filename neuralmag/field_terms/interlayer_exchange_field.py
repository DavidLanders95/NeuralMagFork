# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from neuralmag.common import config
from neuralmag.common.engine import N, Variable, dA
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["InterlayerExchangeField"]


def swap(m, iidx):
    result = config.backend.zeros_like(m)
    result = config.backend.assign(
        result, m[:, :, iidx[1], :], (slice(None), slice(None), iidx[0], slice(None))
    )
    result = config.backend.assign(
        result, m[:, :, iidx[0], :], (slice(None), slice(None), iidx[1], slice(None))
    )
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

        state.iidx = config.backend.tensor(
            self._iidx, device=state.device, dtype=config.backend.integer
        )
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
