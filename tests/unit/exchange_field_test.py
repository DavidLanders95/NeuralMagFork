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

    exchange = ExchangeField()
    h = exchange.h(state).tensor

    exchange_torch = ExchangeTorchField()
    h_torch = exchange_torch.h(state).tensor

    torch.testing.assert_close(h, h_torch, rtol=1e-8, atol=1.0)
