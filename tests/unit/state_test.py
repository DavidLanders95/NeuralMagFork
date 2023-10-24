import pytest
import torch
from neuralmag import *
import numpy as np

def test_tensor_from_list(state):
    t = state.tensor([1,2])
    assert isinstance(t, torch.Tensor)
    assert t.dtype == state.dtype
    assert t.device == state.device

def test_tensor_from_numpy(state):
    t = state.tensor(np.array([1,2]))
    assert isinstance(t, torch.Tensor)
    assert t.dtype == state.dtype
    assert t.device == state.device

def test_setting_float_scalar(state):
    state.a = 1.
    a = state.a
    assert isinstance(a, torch.Tensor)
    state.a = 2
    assert a == 2.

def test_setting_int_scalar(state):
    state.a = 1
    a = state.a
    assert isinstance(a, torch.Tensor)
    assert a.dtype == state.dtype

def test_setting_tensor(state):
    state.a = state.tensor([1.,2.])
    assert state.a.sum().cpu() == pytest.approx(3.)

def test_setting_function(state):
    state.f = Function(state).from_constant(2.)
    assert isinstance(state.f, Function)
    assert state.f.tensor.sum().cpu() == pytest.approx(3**3 * 2.)

def test_dependency_handling(state):
    state.a = 1
    state.b = lambda a: 2*a
    state.c = lambda b: 2*b
    state.material.d = lambda a: 2*a
    assert state.a == 1
    assert state.b == 2
    assert state.c == 4
    assert state.material.d == 2

def test_get_func(state):
    state.a = 1
    state.b = lambda a: 2*a
    c = lambda b: 2*b
    func, args = state.get_func(c)
    assert len(args) == 1
    assert args[0] == state.a

def test_setting_lambda_to_return_function(state):
    state.a = Function(state).from_constant(1.)
    state.f = (lambda a: 2*a, 'node', ())
    assert isinstance(state.f, Function)
    assert state.f.tensor.sum().cpu() == pytest.approx(3**3 * 2.)

def test_wrap_func(state):
    state.a = 1.
    state.b = 2.
    state.c = 3.
    f = lambda a, b: a+b
    g = state.wrap_func(f, {'a': 'c'})
    state.f = f
    state.g = g
    assert state.f.sum().cpu() == pytest.approx(3.)
    assert state.g.sum().cpu() == pytest.approx(5.)
