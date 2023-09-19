import pytest
import torch
from nmagnum import *

def test_scalars():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.a = 1.
    b = state.a
    assert isinstance(b, torch.Tensor)
    state.a = 2.
    assert b == 2.

def test_dependency_handling():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.a = 1
    state.b = lambda a: 2*a
    state.c = lambda b: 2*b
    state.material.d = lambda a: 2*a
    assert state.a == 1
    assert state.b == 2
    assert state.c == 4

    assert state.material.d == 2
