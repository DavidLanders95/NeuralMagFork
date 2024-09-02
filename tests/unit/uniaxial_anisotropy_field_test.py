import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_h(state):
    state.material.Ku = CellFunction(state).fill(1e6)
    state.material.Ku_axis = VectorCellFunction(state).fill([0, 1, 0])
    state.material.Ms = CellFunction(state).fill(8e5)

    UniaxialAnisotropyField().register(state)
    assert be.to_numpy(state.h_uaniso.tensor.sum()) == pytest.approx(23873241.450788274)
