import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_E(state):
    Kc = 1e6
    state.material.Kc = CellFunction(state).fill(Kc)
    state.material.Kc_axis1 = VectorCellFunction(state).fill([1, 0, 0])
    state.material.Kc_axis2 = VectorCellFunction(state).fill([0, 1, 0])
    state.material.Kc_axis3 = VectorCellFunction(state).fill([0, 0, 1])
    state.material.Ms = CellFunction(state).fill(8e5)
    CubicAnisotropyField().register(state)

    state.m = VectorFunction(state).fill((0, 0, 1))

    assert be.to_numpy(state.E_caniso) == pytest.approx(0, abs=1e-25)

    state.m = VectorFunction(state).fill(
        (1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3))
    )
    assert be.to_numpy(state.E_caniso) == pytest.approx(
        -Kc * state.mesh.volume / 3, abs=1e-25
    )


def test_h(state):
    Kc = 1e6
    state.material.Kc = CellFunction(state).fill(Kc)
    state.material.Kc_axis1 = VectorCellFunction(state).fill([1, 0, 0])
    state.material.Kc_axis2 = VectorCellFunction(state).fill([0, 1, 0])
    state.material.Kc_axis3 = VectorCellFunction(state).fill([0, 0, 1])
    state.material.Ms = CellFunction(state).fill(8e5)
    CubicAnisotropyField().register(state)

    assert be.to_numpy(state.h_caniso.tensor).sum() == pytest.approx(6609436.5)
