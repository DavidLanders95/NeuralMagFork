import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_scalar_function(state):
    f = Function(state)
    assert f.tensor.shape == (3, 3, 3)


def test_cell_scalar_function(state):
    f = CellFunction(state)
    assert f.tensor.shape == (2, 2, 2)


def test_vector_function(state):
    f = VectorFunction(state)
    assert f.tensor.shape == (3, 3, 3, 3)


def test_vector_cell_function(state):
    f = VectorCellFunction(state)
    assert f.tensor.shape == (2, 2, 2, 3)


def test_dynamic_function(state):
    state.a = Function(state).fill(2.0)
    f = Function(state, tensor=lambda a: 2 * a)
    assert be.to_numpy(f.avg()) == pytest.approx(4.0)


def test_function_with_mixed_state(state):
    f = Function(state, "ncn")
    assert f.tensor.shape == (3, 2, 3)


def test_vector_function_with_mixed_state(state):
    f = Function(state, "ncn", (3,))
    assert f.tensor.shape == (3, 2, 3, 3)


def test_avg(state):
    f = Function(state).fill(2.0)
    assert be.to_numpy(f.avg()) == pytest.approx(2.0)
    g = CellFunction(state).fill(2.0)
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_fill(state):
    f = Function(state).fill(2.0)
    assert be.to_numpy(f.avg()) == pytest.approx(2.0)


def test_fill_with_vector(state):
    f = VectorFunction(state).fill([1.0, 2.0, 3.0])
    assert be.to_numpy(f.avg()) == pytest.approx([1.0, 2.0, 3.0])


def test_fill_expanded(state):
    f = Function(state).fill(2.0, expand=True)
    assert be.to_numpy(f.avg()) == pytest.approx(2.0)
    assert f.tensor.shape == f.tensor_shape


def test_fill_expanded_with_vector(state):
    f = VectorFunction(state).fill([1.0, 2.0, 3.0], expand=True)
    assert be.to_numpy(f.avg()) == pytest.approx([1.0, 2.0, 3.0])
    assert f.tensor.shape == f.tensor_shape


def test_set_name(state):
    f = CellFunction(state, name="f")
    assert f.name == "f"


def test_tensor_shape(state):
    assert Function(state).tensor_shape == (3, 3, 3)
    assert VectorFunction(state).tensor_shape == (3, 3, 3, 3)
    assert CellFunction(state).tensor_shape == (2, 2, 2)
    assert VectorCellFunction(state).tensor_shape == (2, 2, 2, 3)


def test_fill_by_domain(state):
    x, y, z = state.coordinates()
    state.add_domain(1, x > 1e-9)
    state.add_domain(2, x < 1e-9)

    state.material.A = CellFunction(state).fill_by_domain([0.0, 4.0, 6.0])
    print(state.material.A.func_id)
    assert be.to_numpy(state.material.A.avg()) == pytest.approx(5.0)
