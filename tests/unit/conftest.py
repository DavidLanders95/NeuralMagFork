import numpy as np
import pytest
from scipy import constants

from neuralmag import *

# config.backend = "jax"
config.torch["compile"] = False


@pytest.fixture
def state():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    m_data = np.zeros((3, 3, 3, 3))
    m_data[0, :, :, 0] = -1
    m_data[1, :, :, 1] = 1
    m_data[2, :, :, 0] = 1
    m_data[1, 0, 0, :] = 0
    m_data[1, 0, 0, 0] = 1

    state.m = VectorFunction(state, tensor=state.tensor(m_data))

    state.material.Ms = 1 / constants.mu_0

    return state


@pytest.fixture
def state2d():
    mesh = Mesh((2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    m_data = np.zeros((3, 3, 3))
    m_data[0, :, 0] = -1
    m_data[1, :, 1] = 1
    m_data[2, :, 0] = 1
    state.m = VectorFunction(state, tensor=state.tensor(m_data))

    return state


@pytest.fixture
def state1d():
    mesh = Mesh((2,), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    m_data = np.zeros((3, 3))
    m_data[0, 0] = -1
    m_data[1, 1] = 1
    m_data[2, 0] = 1

    state.m = VectorFunction(state, tensor=state.tensor(m_data))

    return state
