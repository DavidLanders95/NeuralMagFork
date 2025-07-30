import inspect

import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_tensor_from_list(state):
    t = state.tensor([1, 2])
    assert isinstance(t, be.Tensor)
    assert t.dtype == state.dtype
    if be.name == "torch":
        assert t.device == state.device


def test_tensor_from_numpy(state):
    t = state.tensor(np.array([1, 2]))
    assert isinstance(t, be.Tensor)
    assert t.dtype == state.dtype
    if be.name == "torch":
        assert t.device == state.device


def test_setting_float_scalar(state):
    state.a = 1.0
    a = state.a
    assert isinstance(a, be.Tensor)
    state.a = 2.0
    assert state.a == 2.0


def test_setting_int_scalar(state):
    state.a = 1
    a = state.a
    assert isinstance(a, be.Tensor)
    assert a.dtype == state.dtype


def test_setting_tensor(state):
    state.a = state.tensor([1.0, 2.0])
    assert be.to_numpy(state.a.sum()) == pytest.approx(3.0)


def test_setting_function(state):
    state.f = Function(state).fill(2.0)
    assert isinstance(state.f, Function)
    assert be.to_numpy(state.f.tensor.sum()) == pytest.approx(3**3 * 2.0)


def test_dependency_handling(state):
    state.a = 1
    state.b = lambda a: 2 * a
    state.c = lambda b: 2 * b
    state.material.d = lambda a: 2 * a
    assert state.a == 1
    assert state.b == 2
    assert state.c == 4
    assert state.material.d == 2


def test_resolve_with_not_args(state):
    state.a = 1.0
    state.b = lambda a: 2.0 * a
    c = lambda b: 2.0 * b
    func = state.resolve(c)
    arg_names = list(inspect.signature(func).parameters.keys())
    assert arg_names[0] == "a"
    assert func(1.0) == 4.0
    assert func(2.0) == 8.0


def test_resolve(state):
    state.a = 2.0
    state.b = 4.0
    c = lambda a, b: a * b

    func = state.resolve(c, ["a"])
    arg_names = list(inspect.signature(func).parameters.keys())
    assert arg_names[0] == "a"
    assert func(1.0) == 4.0

    func = state.resolve(c, ["b"])
    arg_names = list(inspect.signature(func).parameters.keys())
    assert arg_names[0] == "b"
    assert func(1.0) == 2.0


def test_resolve_overriding_dynamic_attribute(state):
    state.a = 1
    state.b = lambda a: 2 * a
    c = lambda b: 2 * b
    func = state.resolve(c, ["b"])
    arg_names = list(inspect.signature(func).parameters.keys())
    assert arg_names[0] == "b"
    assert func(3.0) == 6.0


def test_resolve_with_inject(state):
    state.a = 2.0
    state.b = 4.0
    c = lambda a, b: a * b

    func = state.resolve(c, ["e"], inject={"b": lambda e: 2 * e})

    arg_names = list(inspect.signature(func).parameters.keys())
    assert arg_names[0] == "e"
    assert func(1.0) == 4.0


def test_setting_lambda_to_return_function(state):
    state.a = Function(state).fill(1.0)
    state.f = (lambda a: 2 * a, "nnn", ())
    assert isinstance(state.f, Function)
    assert be.to_numpy(state.f.tensor.sum()) == pytest.approx(3**3 * 2.0)


def test_remap(state):
    state.a = 1.0
    state.b = 2.0
    state.c = 3.0
    f = lambda a, b: a + b
    g = state.remap(f, {"a": "c"})
    state.f = f
    state.g = g
    assert be.to_numpy(state.f.sum()) == pytest.approx(3.0)
    assert be.to_numpy(state.g.sum()) == pytest.approx(5.0)


def test_remap_func(state):
    state.a = 1.0
    state.b = lambda c: 2.0 * c
    state.c = 3.0
    f = lambda a, b: a + b
    g = state.remap(f, {"b": "c"})
    state.f = f
    state.g = g
    assert be.to_numpy(state.f.sum()) == pytest.approx(7.0)
    assert be.to_numpy(state.g.sum()) == pytest.approx(4.0)


def test_coordinates():
    mesh = Mesh((10, 5, 1), (1e-9, 2e-9, 3e-9), (1e-9, 2e-9, 3e-9))
    state = State(mesh)

    x, y, z = state.coordinates("nnn")
    assert x.shape == (11, 6, 2)
    assert be.to_numpy(x)[1, 0, 0] == pytest.approx(2e-9)
    assert be.to_numpy(y)[0, 1, 0] == pytest.approx(4e-9)
    assert be.to_numpy(z)[0, 0, 0] == pytest.approx(3e-9)

    x, y, z = state.coordinates("ccc")
    assert x.shape == (10, 5, 1)
    assert be.to_numpy(x)[1, 0, 0] == pytest.approx(2.5e-9)
    assert be.to_numpy(y)[0, 1, 0] == pytest.approx(5e-9)
    assert be.to_numpy(z)[0, 0, 0] == pytest.approx(4.5e-9)

    x, y, z = state.coordinates("cnc")
    assert x.shape == (10, 6, 1)
    assert be.to_numpy(x)[1, 0, 0] == pytest.approx(2.5e-9)
    assert be.to_numpy(y)[0, 1, 0] == pytest.approx(4e-9)
    assert be.to_numpy(z)[0, 0, 0] == pytest.approx(4.5e-9)


def test_update_of_tensor(state):
    state.f = state.tensor([1.0])
    state.f_sum = lambda f: f.sum()
    assert be.to_numpy(state.f_sum) == pytest.approx(1.0)
    state.f = state.tensor([2.0])
    assert be.to_numpy(state.f_sum) == pytest.approx(2.0)


def test_update_of_function_tensor(state):
    state.f = Function(state).fill(1.0)
    state.f_sum = lambda f: f.sum()
    assert be.to_numpy(state.f_sum) == pytest.approx(27.0)
    g = Function(state).fill(2.0)
    state.f.tensor = g.tensor
    assert be.to_numpy(state.f_sum) == pytest.approx(54.0)


def test_add_domain(state):
    x, y, z = state.coordinates()
    assert be.to_numpy(state.rho.avg()) == pytest.approx(1.0)
    state.add_domain(1, x > 1e-9)
    state.add_domain(2, x < 1e-9)
    assert be.to_numpy(state.m.avg(1)) == pytest.approx([0.5312498, 0.46875, 0.0])
    assert be.to_numpy(state.m.avg(2)) == pytest.approx([-0.46875, 0.46875, 0.0])
    assert be.to_numpy(state.m.avg(1) + state.m.avg(2)) / 2.0 == pytest.approx(
        be.to_numpy(state.m.avg()), abs=1e-7
    )
