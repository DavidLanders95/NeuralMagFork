import numpy as np
import pytest
import torch

from nmagnum import *


def test_h():
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.m = VectorFunction(state).from_constant([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).from_constant(1.0)

    demag = DemagField(state)
    h = demag.h()
    assert ((h.tensor * state.m.tensor).sum() / 6**3).cpu().numpy() == pytest.approx(
        -1 / 3
    )
