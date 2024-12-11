import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_h(state):
    state.material.A = CellFunction(state).fill(1.2e-11)
    state.material.Di = CellFunction(state).fill(1e-3)
    state.material.Di_axis = VectorCellFunction(state).fill([0, 0, 1])
    state.material.Ms = CellFunction(state).fill(8e5)

    ExchangeField().register(state)
    InterfaceDMIField().register(state)
    TotalField("exchange", "idmi").register(state)
    assert be.to_numpy(state.h.tensor.sum()) == pytest.approx(448728519.86203885)
