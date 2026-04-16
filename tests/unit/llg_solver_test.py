import pytest

from neuralmag import *

be = config.backend


# When running with the JAX backend, exercise all supported solver types.
# Otherwise, keep a single default case (None) to use backend defaults.
if be.name == "jax":
    _SOLVER_TYPES = ["Dopri5", "Kvaerno3", "Kvaerno5", "KenCarp3", "KenCarp5"]
else:
    _SOLVER_TYPES = [None]


@pytest.mark.parametrize("solver_type", _SOLVER_TYPES)
def test_step(state, solver_type):
    state.material.alpha = 1.0
    state.m.fill([1, 0, 0])

    # regist h for implicit and explicit solvers
    h_ext = VectorFunction(state).fill([0, 0, 8e5])
    ExternalField(h_ext).register(state, "")

    # register h_impl and h_expl for imex solvers
    h_ext_half = VectorFunction(state).fill([0, 0, 4e5])
    ExternalField(h_ext_half).register(state, "impl")
    ExternalField(h_ext_half).register(state, "expl")

    if solver_type is None:
        llg = LLGSolver(state)
    else:
        llg = LLGSolver(state, solver_type=solver_type)

    assert be.to_numpy(state.m.avg()) == pytest.approx((1, 0, 0))
    llg.step(1e-11)
    assert be.to_numpy(state.m.avg()) == pytest.approx((0.44655174, 0.54584754, 0.70895207), rel=1e-3)


def test_torch_parameters(state):
    if be.name == "jax":
        pytest.skip()

    state.material.alpha = 1.0
    state.m.fill([1, 0, 0])
    h_ext = VectorFunction(state).fill([0, 0, 8e5])
    ExternalField(h_ext).register(state, "")

    state.a = Function(state).fill(0.0)

    llg = LLGSolver(state, parameters=["a"])
    assert len(list(llg.parameters())) == 1
