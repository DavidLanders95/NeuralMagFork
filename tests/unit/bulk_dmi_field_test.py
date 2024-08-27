import numpy as np
import pytest

from neuralmag import *


def test_h(state):
    state.material.Db = CellFunction(state).fill(1e-3)
    state.material.Ms = CellFunction(state).fill(8e5)

    BulkDMIField().register(state)
    assert state.h_bdmi.tensor.sum().cpu() == pytest.approx(4.656612873077393e-10)
