import pytest

from neuralmag import *

be = config.backend


def test_h(state):
    state.material.Di = CellFunction(state).fill(1e-3)
    state.material.Di_axis = VectorCellFunction(state).fill([0, 0, 1])
    state.material.Ms = CellFunction(state).fill(8e5)

    InterfaceDMIField().register(state)
    assert be.to_numpy(state.h_idmi.tensor.sum()) == pytest.approx(19010173.74784992)
