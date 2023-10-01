import numpy as np
import pytest
import torch
from nmagnum import *

def test_h(state):
    state.material.A = CellFunction(state).from_constant(1.2e-11)
    state.material.Ms = CellFunction(state).from_constant(8e5)

    ExchangeField().register(state)
    assert state.h_exchange.tensor.sum().cpu() == pytest.approx(429718346.1141888)
