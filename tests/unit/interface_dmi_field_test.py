import numpy as np
import pytest
from neuralmag import *

def test_h(state):
    state.material.Di = CellFunction(state).from_constant(1e-3)
    state.material.Di_axis = VectorCellFunction(state).from_constant([0,0,1])
    state.material.Ms = CellFunction(state).from_constant(8e5)

    InterfaceDMIField().register(state)
    assert state.h_idmi.tensor.sum().cpu() == pytest.approx(17904931.088091202)
