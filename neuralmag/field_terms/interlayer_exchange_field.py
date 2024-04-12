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
    _name = "iexchange"

    def __init__(self, idx1, idx2, *args, **kwargs):
        super().__init__(**kwargs)
        self._iidx = [idx1, idx2]

    def register(self, state, name=None):
        super().register(state, name)

        # state.iidx = state.tensor(self._iidx, dtype = torch.int)
        state.iidx = torch.tensor(self._iidx, device=state.device, dtype=torch.int)
        state.im_other = swap

    @staticmethod
    def e_expr(m, dim):
        iA = Variable("material__iA", "node", dim)
        im_other = Variable("im_other", "node", dim, (3,))

        return -0.5 * iA * m.dot(im_other) * dA(normal=2, idx="iidx")
