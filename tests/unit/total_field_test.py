import numpy as np
import pytest
import torch

from neuralmag import *


def test_h(state):
    state.material.A = CellFunction(state).fill(1.2e-11)
    state.material.Di = CellFunction(state).fill(1e-3)
    state.material.Di_axis = VectorCellFunction(state).fill([0, 0, 1])
    state.material.Ms = CellFunction(state).fill(8e5)

    ExchangeField().register(state)
    InterfaceDMIField().register(state)
    TotalField("exchange", "idmi").register(state)
    assert state.h.tensor.sum().cpu() == pytest.approx(447623277.2022799)
