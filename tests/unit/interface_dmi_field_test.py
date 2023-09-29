import numpy as np
import pytest
from nmagnum import *

def test_h():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_numpy(np.arange(81).reshape(3, 3, 3, 3))
    state.material.Di = CellFunction(state).from_constant(1e-3)
    state.material.Di_axis = VectorCellFunction(state).from_constant([0,0,1])
    state.material.Ms = CellFunction(state).from_constant(8e5)

    InterfaceDMIField().register(state)
    assert state.h_idmi.tensor.sum().cpu() == pytest.approx(-2.086162567138672e-07)
