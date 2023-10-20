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

def test_from_constant(state):
    f = Function(state).from_constant(2.0)
    assert f.avg().cpu() == pytest.approx(2.0)

def test_from_constant_with_vector(state):
    f = VectorFunction(state).from_constant([1.0, 2.0, 3.0])
    assert f.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])

def test_set_name(state):
    f = CellFunction(state, name = 'f')
    assert f.name == 'f'
