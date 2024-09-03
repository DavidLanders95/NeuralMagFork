import numpy as np
import pytest
from scipy import constants

from neuralmag import *

be = config.backend


def test_h():
    mesh = Mesh((2, 2, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    m_data = np.zeros((3, 3, 6, 3))
    m_data[:, :, :3, 0] = 1.0
    m_data[:, :, 3:, 1] = 1.0

    state.m = VectorFunction(state, tensor=state.tensor(m_data))

    state.material.Ms = 1 / constants.mu_0
    state.material.iA = Function(state, "ccn").fill(0.005, expand=True)

    rho_data = np.ones(mesh.n)
    rho_data[:, :, 2] = state.eps

    state.rho = CellFunction(state, tensor=state.tensor(rho_data))

    InterlayerExchangeField(2, 3).register(state)
    assert be.to_numpy(state.h_iexchange.tensor[:, :, 3, :]).mean(
        axis=(0, 1)
    ) == pytest.approx([2 * 0.005 / 1e-9, 0, 0])
    assert be.to_numpy(state.h_iexchange.tensor[:, :, 2, :]).mean(
        axis=(0, 1)
    ) == pytest.approx([0, 2 * 0.005 / 1e-9, 0])
