from scipy import constants
from ..common import Function, VectorFunction
from .field_term import FieldTerm
from .cache.exchange import assemble_linear_form

__all__ = ["ExchangeTorchField2"]


class ExchangeTorchField2(FieldTerm):
    def __init__(self):
        self._state = None

    def h(self, state):
        if self._state is None:
            self._state = state
            self._h = VectorFunction(state)

        assemble_linear_form(self._h.tensor, state.mesh.dx, state.m.tensor, state.material.A.tensor)

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


