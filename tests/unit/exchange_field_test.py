import pytest

from neuralmag import *

be = config.backend


def test_h(state):
    state.material.A = CellFunction(state).fill(1.2e-11)
    state.material.Ms = CellFunction(state).fill(8e5)

    ExchangeField().register(state)
    assert be.to_numpy(state.h_exchange.tensor.sum()) == pytest.approx(
        429718346.1141888
    )


def test_h_2d(state2d):
    state2d.material.A = CellFunction(state2d).fill(1.2e-11)
    state2d.material.Ms = CellFunction(state2d).fill(8e5)

    ExchangeField().register(state2d)
    assert be.to_numpy(state2d.h_exchange.tensor.sum()) == pytest.approx(
        143239448.70472956
    )


def test_h_other_name(state):
    state.material.A = CellFunction(state).fill(1.2e-11)
    state.material.Ms = CellFunction(state).fill(8e5)

    ExchangeField().register(state, "ex")
    assert be.to_numpy(state.h_ex.tensor.sum()) == pytest.approx(429718346.1141888)
