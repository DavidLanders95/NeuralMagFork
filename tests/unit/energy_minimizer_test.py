import numpy as np
import pytest
from scipy import constants

from neuralmag import *

be = config.backend


def _normalized(state, value):
    vector = state.tensor(value)
    return vector / be.np.linalg.norm(vector)


def _make_uniaxial_state():
    state = State(Mesh((1, 1, 1), (2e-9, 2e-9, 2e-9)))
    state.material.Ms = 8e5
    state.material.Ku = 2e5
    state.material.Ku_axis = [0, 0, 1]
    state.material.alpha = 1.0
    state.m = VectorFunction(state).fill(_normalized(state, [1.0, 0.0, 0.2]))
    UniaxialAnisotropyField().register(state, "")
    return state


def test_minimize_aligns_with_external_field():
    state = State(Mesh((1, 1, 1), (2e-9, 2e-9, 2e-9)))
    state.material.Ms = 8e5
    state.m = VectorFunction(state).fill([0.0, 0.0, 1.0])

    h_ext = VectorFunction(state).fill([8e5, 0.0, 0.0])
    ExternalField(h_ext).register(state, "")

    minimizer = EnergyMinimizer(state, tol=1e-2, max_iter=200)
    minimizer.minimize()

    assert be.to_numpy(state.m.avg()) == pytest.approx((1.0, 0.0, 0.0), abs=1e-3)

    expected_energy = -constants.mu_0 * 8e5 * 8e5 * state.mesh.volume
    assert be.to_numpy(state.E) == pytest.approx(expected_energy, rel=1e-4)


def test_minimize_matches_llg_relaxation_for_uniaxial_state():
    state_min = _make_uniaxial_state()
    state_llg = _make_uniaxial_state()

    minimizer = EnergyMinimizer(state_min, tol=1e-2, max_iter=200)
    minimizer.minimize()

    llg = LLGSolver(state_llg)
    llg.relax(tol=1e7)

    assert be.to_numpy(state_min.m.avg()) == pytest.approx(be.to_numpy(state_llg.m.avg()), abs=1e-3)
    assert be.to_numpy(state_min.E) == pytest.approx(be.to_numpy(state_llg.E), rel=1e-4)