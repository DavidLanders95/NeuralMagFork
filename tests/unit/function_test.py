import numpy as np
import pytest

from nmagnum import *


@pytest.fixture
def simple_state():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    return State(mesh)


def test_scalar_function(simple_state):
    f = Function(simple_state)
    assert f.tensor.shape == (6, 6, 6)


def test_cell_scalar_function(simple_state):
    f = CellFunction(simple_state)
    assert f.tensor.shape == (5, 5, 5)


def test_vector_function(simple_state):
    f = VectorFunction(simple_state)
    assert f.tensor.shape == (6, 6, 6, 3)


def test_vector_cell_function(simple_state):
    f = VectorCellFunction(simple_state)
    assert f.tensor.shape == (5, 5, 5, 3)


def test_from_constant(simple_state):
    f = Function(simple_state).from_constant(2.0)
    assert f.avg().cpu() == pytest.approx(2.0)


def test_from_constant_with_vector(simple_state):
    f = VectorFunction(simple_state).from_constant([1.0, 2.0, 3.0])
    assert f.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_from_numpy(simple_state):
    f = CellFunction(simple_state).from_numpy(
        np.arange(simple_state.mesh.num_cells).reshape(simple_state.mesh.n)
    )
    assert f.avg().cpu() == pytest.approx(np.arange(simple_state.mesh.num_cells).mean())
