import numpy as np
import pytest
from neuralmag import *

def test_h(state):
    state.material.Ku = CellFunction(state).from_constant(1e6)
    state.material.Ku_axis = VectorCellFunction(state).from_constant([0,1,0])
    state.material.Ms = CellFunction(state).from_constant(8e5)

    UniaxialAnisotropyField().register(state)
    assert state.h_uaniso.tensor.sum().cpu() == pytest.approx(23873241.450788274)
