import numpy as np
import pytest
from scipy import constants

from neuralmag import *

config.torch["compile"] = False


@pytest.fixture
def state():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state)
    state.m.tensor[0, :, :, 0] = -1
    state.m.tensor[1, :, :, 1] = 1
    state.m.tensor[2, :, :, 0] = 1

    state.material.Ms = 1 / constants.mu_0

    return state


@pytest.fixture
def state2d():
    mesh = Mesh((2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state)
    state.m.tensor[0, :, 0] = -1
    state.m.tensor[1, :, 1] = 1
    state.m.tensor[2, :, 0] = 1

    return state


@pytest.fixture
def state1d():
    mesh = Mesh((2,), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state)
    state.m.tensor[0, 0] = -1
    state.m.tensor[1, 1] = 1
    state.m.tensor[2, 0] = 1

    return state
