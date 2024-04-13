import numpy as np
import pytest

from neuralmag import *


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


def test_function_with_mixed_state(state):
    f = Function(state, "ncn")
    assert f.tensor.shape == (3, 2, 3)


def test_vector_function_with_mixed_state(state):
    f = Function(state, "ncn", (3,))
    assert f.tensor.shape == (3, 2, 3, 3)


def test_fill(state):
    f = Function(state).fill(2.0)
    assert f.avg().cpu() == pytest.approx(2.0)


def test_fill_with_vector(state):
    f = VectorFunction(state).fill([1.0, 2.0, 3.0])
    assert f.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_expand(state):
    f = Function(state).expand(2.0)
    assert f.avg().cpu() == pytest.approx(2.0)
    assert f.tensor.size() == f._size


def test_expand_with_vector(state):
    f = VectorFunction(state).expand([1.0, 2.0, 3.0])
    assert f.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])
    assert f.tensor.size() == f._size


def test_set_name(state):
    f = CellFunction(state, name="f")
    assert f.name == "f"
