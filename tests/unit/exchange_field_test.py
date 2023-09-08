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

    #exchange = ExchangeField()
    #h = exchange.h(state).tensor

    exchange = ExchangeField(state)
    h_torch = exchange.h().tensor

    exchange_torch2 = ExchangeTorchField2()
    h_torch2 = exchange_torch2.h(state).tensor

    #torch.testing.assert_close(h, h_torch, rtol=1e-8, atol=1.0)
    torch.testing.assert_close(h_torch, h_torch2, rtol=1e-8, atol=1.0)

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
