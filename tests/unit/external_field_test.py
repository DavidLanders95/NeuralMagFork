import numpy as np
import pytest

from neuralmag import *


def test_h_from_field(state):
    h_ext = VectorFunction(state).fill([1.0, 2.0, 3.0])
    ExternalField(h_ext).register(state)
    assert state.h_external.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_h_from_array(state):
    ExternalField(state.tensor([1.0, 2.0, 3.0])).register(state)
    assert state.h_external.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_h_from_func(state):
    h_ext = lambda t: t * state.tensor([1.0, 2.0, 3.0]).expand((3, 3, 3, 3))
    ExternalField(h_ext).register(state)
    state.t = 0.0
    assert state.h_external.avg().cpu() == pytest.approx([0.0, 0.0, 0.0])
    state.t = 1.0
    assert state.h_external.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_h_from_array_func(state):
    h_ext = lambda t: t * state.tensor([1.0, 2.0, 3.0])
    ExternalField(h_ext).register(state)
    state.t = 0.0
    assert state.h_external.avg().cpu() == pytest.approx([0.0, 0.0, 0.0])
    state.t = 1.0
    assert state.h_external.avg().cpu() == pytest.approx([1.0, 2.0, 3.0])


def test_E(state):
    h_ext = VectorFunction(state).fill([1.0, 2.0, 3.0])
    ExternalField(h_ext).register(state)
    assert state.E_external.cpu() == pytest.approx(-8e-27)
