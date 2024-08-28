import numpy as np
import pytest

from neuralmag import *


def test_h(state):
    state.material.Db = CellFunction(state).fill(1e-3)
    state.material.Ms = CellFunction(state).fill(8e5)
    state.m.tensor[1, 0, 0, :] = 0.0
    state.m.tensor[1, 0, 0, 0] = 1.0

    BulkDMIField().register(state)
    assert state.h_bdmi.tensor.sum().cpu() == pytest.approx(1105242.0)
