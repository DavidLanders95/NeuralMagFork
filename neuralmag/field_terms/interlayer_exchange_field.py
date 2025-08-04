# SPDX-License-Identifier: MIT

from neuralmag.common import VectorFunction, config
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
        # state.im_other = (swap, "node", (3,))
        state.im_other = VectorFunction(state, tensor=swap)

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
