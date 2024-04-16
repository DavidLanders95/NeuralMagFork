import numpy as np
import pytest
import torch

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


def test_fill_expanded(state):
    f = Function(state).fill(2.0, expand=True)
    assert f.avg().cpu() == pytest.approx(2.0)
    assert f.tensor.size() == f._size
    f.tensor[0, 0, 0] = 3.0
    assert f.avg().cpu() == pytest.approx(3.0)


def test_fill_expanded_with_vector(state):
    f = VectorFunction(state).fill([1.0, 2.0, 3.0], expand=True)
    assert f.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])
    assert f.tensor.size() == f._size
    f.tensor[0, 0, 0, :] = torch.tensor([4.0, 5.0, 6.0])
    assert f.avg().cpu() == pytest.approx([4.0, 5.0, 6.0])


def test_set_name(state):
    f = CellFunction(state, name="f")
    assert f.name == "f"


def test_size(state):
    assert Function(state).size == (3, 3, 3)
    assert VectorFunction(state).size == (3, 3, 3, 3)
    assert CellFunction(state).size == (2, 2, 2)
    assert VectorCellFunction(state).size == (2, 2, 2, 3)
