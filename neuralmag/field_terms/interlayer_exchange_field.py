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

    where :math:`\Gamma` denotes two coupled interfaces and :math:`\vec{m}_\text{other}` denotes the magnetization at the nearest point of the other interface. This expression is equivalent to the more common expression

    .. math::

      E = - \int_\Gamma A \vec{m}_1 \cdot \vec{m}_2 \ds

    where :math:`\Gamma` denotes a shared interface and :math:`\vec{m}_1` and :math:`\vec{m}_2` denote the magnetization on the respective sides.
    """
    _name = "iexchange"

    def __init__(self, idx1, idx2, *args, **kwargs):
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
