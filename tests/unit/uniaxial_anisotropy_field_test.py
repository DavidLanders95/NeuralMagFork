import numpy as np
import pytest
from nmagnum import *

def test_h():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_numpy(np.arange(81).reshape(3, 3, 3, 3))
    state.material.Ku = CellFunction(state).from_constant(1e6)
    state.material.Ku_axis = VectorCellFunction(state).from_constant([0,0,1])
    state.material.Ms = CellFunction(state).from_constant(8e5)

    UniaxialAnisotropyField().register(state)
    assert state.h_uaniso.tensor.sum().cpu() == pytest.approx(2202306523.8352184)
