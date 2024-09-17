import numpy as np
import pytest
import torch

from neuralmag import *

be = config.backend


def test_step(state):
    state.material.alpha = 1.0
    state.m.fill([1, 0, 0])

    h_ext = VectorFunction(state).fill([0, 0, 8e5])
    ExternalField(h_ext).register(state, "")

    llg = LLGSolver(state)

    assert be.to_numpy(state.m.avg()) == pytest.approx((1, 0, 0))
    llg.step(1e-11)
    assert be.to_numpy(state.m.avg()) == pytest.approx(
        (0.44655174, 0.54584754, 0.70895207)
    )
