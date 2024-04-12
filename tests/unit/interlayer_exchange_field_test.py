import numpy as np
import pytest
import torch
from scipy import constants

from neuralmag import *


def test_h():
    mesh = Mesh((2, 2, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state)
    state.m.tensor[:, :, :3, 0] = 1
    state.m.tensor[:, :, 3:, 1] = 1

    state.material.iA = Function(state).from_constant(0.005)
    state.material.Ms = 1 / constants.mu_0
    state.rho = CellFunction(state).from_constant(1.0)
    state.rho.tensor[:, :, 2] = torch.finfo(state.dtype).eps

    InterlayerExchangeField(2, 3).register(state)
    assert state.h_iexchange.tensor[:, :, 3, :].mean(dim=(0, 1)).cpu() == pytest.approx(
        [0.005 / 1e-9, 0, 0]
    )
    assert state.h_iexchange.tensor[:, :, 2, :].mean(dim=(0, 1)).cpu() == pytest.approx(
        [0, 0.005 / 1e-9, 0]
    )
