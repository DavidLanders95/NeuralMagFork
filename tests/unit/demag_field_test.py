import pytest
import torch

from neuralmag import *

be = config.backend


def test_h():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)

    DemagField().register(state)
    assert be.to_numpy(
        (state.h_demag.tensor * state.m.tensor).sum() / 6**3
    ) == pytest.approx(-1 / 3)


def test_h_2d():
    mesh = Mesh((5, 5), (1e-9, 1e-9, 5e-9))
    state = State(mesh)

    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)

    DemagField().register(state)
    assert be.to_numpy(
        (state.h_demag.tensor * state.m.tensor).sum() / 6**2
    ) == pytest.approx(-1 / 3, 1e-1)


def test_E():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)

    DemagField().register(state)
    assert be.to_numpy(state.E_demag) == pytest.approx(2.6179938794166683e-32)


def test_rename():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)

    DemagField().register(state, "d")
    assert be.to_numpy(
        (state.h_d.tensor * state.m.tensor).sum() / 6**3
    ) == pytest.approx(-1 / 3)
    assert be.to_numpy(state.E_d) == pytest.approx(2.6179938794166683e-32)
