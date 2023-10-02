import numpy as np
import pytest
import torch

from nmagnum import *

def test_h(state):
    state.material.A = CellFunction(state).from_constant(1.2e-11)
    state.material.Di = CellFunction(state).from_constant(1e-3)
    state.material.Di_axis = VectorCellFunction(state).from_constant([0,0,1])
    state.material.Ms = CellFunction(state).from_constant(8e5)

    ExchangeField().register(state)
    InterfaceDMIField().register(state)
    TotalField('exchange', 'idmi').register(state)
    assert state.h.tensor.sum().cpu() == pytest.approx(447623277.2022799)
