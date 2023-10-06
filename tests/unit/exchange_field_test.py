import numpy as np
import pytest
import torch
from nmagnum import *


def test_h(state):
    state.material.A = CellFunction(state).from_constant(1.2e-11)
    state.material.Ms = CellFunction(state).from_constant(8e5)

    ExchangeField().register(state)
    assert state.h_exchange.tensor.sum().cpu() == pytest.approx(429718346.1141888)

def test_h_2d(state2d):
    state2d.material.A = CellFunction(state2d).from_constant(1.2e-11)
    state2d.material.Ms = CellFunction(state2d).from_constant(8e5)

    ExchangeField().register(state2d)
    assert state2d.h_exchange.tensor.sum().cpu() == pytest.approx(143239448.70472956)

def test_h_other_name(state):
    state.material.A = CellFunction(state).from_constant(1.2e-11)
    state.material.Ms = CellFunction(state).from_constant(8e5)

    ExchangeField().register(state, 'ex')
    assert state.h_ex.tensor.sum().cpu() == pytest.approx(429718346.1141888)
