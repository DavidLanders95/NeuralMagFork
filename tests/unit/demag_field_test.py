import pytest
import torch
from nmagnum import *

def test_h():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_constant([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).from_constant(1.0)

    DemagField().register(state)
    assert ((state.h_demag.tensor * state.m.tensor).sum() / 6**3).cpu() == pytest.approx(-1 / 3)

def test_E():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_constant([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).from_constant(1.0)

    DemagField().register(state)
    assert state.E_demag.cpu() == pytest.approx(2.6179938794166683e-32)

def test_rename():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_constant([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).from_constant(1.0)

    DemagField().register(state, 'd')
    assert ((state.h_d.tensor * state.m.tensor).sum() / 6**3).cpu() == pytest.approx(-1 / 3)
    assert state.E_d.cpu() == pytest.approx(2.6179938794166683e-32)
