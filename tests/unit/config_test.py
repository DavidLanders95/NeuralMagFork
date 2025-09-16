import pytest

from neuralmag import *

be = config.backend


def test_dtype_default():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.x = Function(state)
    assert state.x.tensor.dtype == be.float32


def test_dtype_default_single():
    config.dtype = "float32"
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.x = Function(state)
    assert state.x.tensor.dtype == be.float32


def test_dtype_default_double():
    config.dtype = "float64"
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.x = Function(state)
    assert state.x.tensor.dtype == be.float64


def test_dtype_state_single():
    if be.name == "jax":
        pytest.skip()
    config.dtype = "float64"
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh, dtype="float32")
    state.x = Function(state)
    assert state.x.tensor.dtype == be.float32


def test_dtype_state_double():
    if be.name == "jax":
        pytest.skip()
    config.dtype = "float32"
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh, dtype="float64")
    state.x = Function(state)
    assert state.x.tensor.dtype == be.float64
