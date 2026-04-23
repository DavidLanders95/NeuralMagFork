import numpy as np
import pytest
from scipy import constants

from neuralmag import *

be = config.backend


def _normalized(state, value):
    vector = state.tensor(value)
    return vector / be.np.linalg.norm(vector)


def _make_uniaxial_state(vector=(1.0, 0.0, 0.2)):
    state = State(Mesh((1, 1, 1), (2e-9, 2e-9, 2e-9)))
    state.material.Ms = 8e5
    state.material.Ku = 2e5
    state.material.Ku_axis = [0, 0, 1]
    state.material.alpha = 1.0
    state.m = VectorFunction(state).fill(_normalized(state, vector))
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


def test_jax_minimize_return_info_reports_early_stop():
    if be.name != "jax":
        pytest.skip()

    state = _make_uniaxial_state([0.0, 0.0, 1.0])
    solver = EnergyMinimizer(state, tol=1e-6, max_iter=10)

    max_g, info = solver.minimize(return_info=True)

    assert np.asarray(be.to_numpy(info["converged"])).item() is True
    assert np.asarray(be.to_numpy(info["n_iter"])).item() == 0
    assert be.to_numpy(info["max_g"]) == pytest.approx(be.to_numpy(max_g), abs=1e-12)


def test_jax_reset_clears_history_without_changing_result():
    if be.name != "jax":
        pytest.skip()

    state = _make_uniaxial_state()
    solver = EnergyMinimizer(state, tol=1e-2, max_iter=200)
    solver.step()
    solver.reset()

    _, info = solver.minimize(return_info=True)

    assert solver.n_iter == np.asarray(be.to_numpy(info["n_iter"])).item()
    assert np.asarray(be.to_numpy(info["converged"])).item() is True
    assert be.to_numpy(state.m.avg()) == pytest.approx((0.0, 0.0, 1.0), abs=1e-3)


def test_minimize_many_matches_looped_uniaxial_solves():
    if be.name != "jax":
        pytest.skip()

    vectors = ([1.0, 0.0, 0.2], [-1.0, 0.0, -0.1])
    template_state = _make_uniaxial_state(vectors[0])
    solver = EnergyMinimizer(template_state, tol=1e-2, max_iter=200)

    m_batch = be.np.stack([_make_uniaxial_state(vector).m.tensor for vector in vectors], axis=0)
    m_final, info = solver.minimize_many(m_batch, return_info=True)

    looped_states = []
    looped_iterations = []
    for vector in vectors:
        state = _make_uniaxial_state(vector)
        single = EnergyMinimizer(state, tol=1e-2, max_iter=200)
        single.minimize()
        looped_states.append(be.to_numpy(state.m.tensor))
        looped_iterations.append(single.n_iter)

    assert np.asarray(be.to_numpy(info["converged"])).tolist() == [True, True]
    assert be.to_numpy(m_final) == pytest.approx(np.stack(looped_states, axis=0), abs=1e-6)
    assert be.to_numpy(info["n_iter"]) == pytest.approx(np.asarray(looped_iterations))


def test_minimize_many_can_reach_distinct_uniaxial_minima():
    if be.name != "jax":
        pytest.skip()

    template_state = _make_uniaxial_state([1.0, 0.0, 0.2])
    solver = EnergyMinimizer(template_state, tol=1e-2, max_iter=200)
    m_batch = be.np.stack(
        [
            _make_uniaxial_state([1.0, 0.0, 0.2]).m.tensor,
            _make_uniaxial_state([-1.0, 0.0, -0.2]).m.tensor,
        ],
        axis=0,
    )

    m_final, info = solver.minimize_many(m_batch, return_info=True)
    avg_z = np.asarray(be.to_numpy(m_final))[..., 2].reshape(2, -1).mean(axis=-1)

    assert np.asarray(be.to_numpy(info["converged"])).tolist() == [True, True]
    assert avg_z[0] == pytest.approx(1.0, abs=1e-6)
    assert avg_z[1] == pytest.approx(-1.0, abs=1e-6)
