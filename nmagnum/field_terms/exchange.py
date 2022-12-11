from ..common import VectorFunction
from .field_term import FieldTerm
from .kernels.exchange import kernel, mesh_d, mesh_N

__all__ = ["ExchangeField"]


class ExchangeField(FieldTerm):
    def __init__(self):
        self._state = None

    def h(self, state):
        if self._state is None:
            self._state = state
            self._h = VectorFunction(state)

            # TODO this has to be optimized
            shape = state.mesh.n
            self._tpb = (8, 8, 8)
            _bpg = []
            for i in range(3):
                _bpg.append((shape[i] + (self._tpb[i] - 1)) // self._tpb[i])

            self._bpg = tuple(_bpg)

            # fill constant values for CUDA kernel
            mesh_d[:] = state.mesh.dx
            mesh_N[:] = state.mesh.n

        kernel[self._bpg, self._tpb](
            state.m.tensor,
            state.material.A.tensor,
            state.material.Ms.tensor,
            self._h.tensor,
        )

        return self._h
