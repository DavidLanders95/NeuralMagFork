import numpy as np
import pytest
import torch

from nmagnum import *

def test_h():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    state = State(mesh)

    state.a = VectorFunction(state).from_constant((1., 2., 3.))
    state.b = VectorFunction(state).from_constant((4., 5., 6.))
    TotalField('a', 'b').register(state)
    torch.testing.assert_close(state.h_total.tensor.sum(), state.tensor(567.))

