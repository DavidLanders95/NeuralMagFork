import numpy as np
import pytest
import torch

from nmagnum import *

def test_h():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_numpy(np.arange(81).reshape(3, 3, 3, 3))
    state.material.A = CellFunction(state).from_constant(1.2e-11)
    state.material.Ms = CellFunction(state).from_constant(8e5)

    exchange = ExchangeField(state)
    h = exchange.h().tensor

    assert h.sum() == pytest.approx(3.0994415283203125e-06)

#def test_E():
#    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
#    state = State(mesh)
#
#    state.m = VectorFunction(state).from_numpy(np.arange(81).reshape(3, 3, 3, 3))
#    state.material.A = CellFunction(state).from_constant(1.2e-11)
#    state.material.Ms = CellFunction(state).from_constant(8e5)
#
#    exchange_torch2 = ExchangeTorchField2()
#    assert exchange_torch2.E(state) == pytest.approx(2.3587200000000006e-16)
