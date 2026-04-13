# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from neuralmag import *

be = config.backend


# ===========================================================================
# Exchange field PBC tests
# ===========================================================================


class TestExchangeFEMPBC:
    """Exchange field FEM PBC tests using two-domain configuration."""

    @staticmethod
    def _make_state(pbc=(0, 0, 0)):
        mesh = Mesh((10, 2, 2), (5e-9, 5e-9, 5e-9), pbc=pbc)
        state = State(mesh)
        n_x = 10 if pbc[0] else 11
        n_y = 2 if pbc[1] else 3
        n_z = 2 if pbc[2] else 3
        m_data = np.zeros((n_x, n_y, n_z, 3))
        m_data[: n_x // 2, :, :, 2] = 1
        m_data[n_x // 2 :, :, :, 2] = -1
        state.m = VectorFunction(state, tensor=state.tensor(m_data))
        state.material.A = CellFunction(state).fill(1.3e-11)
        state.material.Ms = CellFunction(state).fill(8e5)
        return state

    def test_pbc_antisymmetric(self):
        """With full PBC, h[:5] == -h[5:]."""
        state = self._make_state(pbc=(1, 1, 1))
        ExchangeField().register(state)
        h = be.to_numpy(state.h_exchange.tensor)
        np.testing.assert_allclose(h[:5], -h[5:], atol=1e-10)

    def test_pbc_vs_open(self):
        """Interior matches; boundaries differ."""
        state_pbc = self._make_state(pbc=(1, 0, 0))
        state_open = self._make_state(pbc=(0, 0, 0))
        ExchangeField().register(state_pbc)
        ExchangeField().register(state_open)

        h_pbc = be.to_numpy(state_pbc.h_exchange.tensor)
        h_open = be.to_numpy(state_open.h_exchange.tensor)

        # Interior domain-wall nodes match
        np.testing.assert_allclose(h_pbc[4], h_open[4], atol=1e-10)
        np.testing.assert_allclose(h_pbc[5], h_open[5], atol=1e-10)

        # Boundary nodes differ
        assert np.abs(h_open[0] - h_pbc[0]).max() > 0
        assert np.abs(h_open[-1] - h_pbc[-1]).max() > 0


class TestExchangeFICPBC:
    """Exchange field FIC (cell-centred) PBC tests."""

    @staticmethod
    def _make_state(pbc=(0, 0, 0)):
        mesh = Mesh((10, 2, 2), (5e-9, 5e-9, 5e-9), pbc=pbc)
        state = State(mesh)
        m_data = np.zeros((10, 2, 2, 3))
        m_data[:5, :, :, 2] = 1
        m_data[5:, :, :, 2] = -1
        state.m = VectorCellFunction(state, tensor=state.tensor(m_data))
        state.material.A = CellFunction(state).fill(1.3e-11)
        state.material.Ms = CellFunction(state).fill(8e5)
        return state

    def test_pbc_antisymmetric(self):
        state = self._make_state(pbc=(1, 1, 1))
        ExchangeField().register(state)
        h = be.to_numpy(state.h_exchange.tensor)
        np.testing.assert_allclose(h[:5], -h[5:], atol=1e-10)


# ===========================================================================
# Demag field PBC tests
# ===========================================================================


def test_demag_cell_pbc_uniform():
    """For uniform magnetization with full PBC, h_demag should be zero."""
    mesh = Mesh(
        (5, 5, 5),
        (1e-9, 1e-9, 1e-9),
        pbc=True,
    )
    state = State(mesh)
    state.m = VectorCellFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)

    h = be.to_numpy(state.h_demag.tensor)
    assert h[..., 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 1].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 2].mean() == pytest.approx(0.0, abs=1e-6)


def test_demag_cell_pbc_stripes():
    """Periodic stripes from Bruckner et al., Sci. Rep. 11, 9202 (2021)."""
    d1, d0 = 4, 6
    nx = d1 + d0
    mesh = Mesh(
        (nx, 2, 2),
        (1e-9, 1e-9, 1e-9),
        pbc=True,
    )
    state = State(mesh)
    state.m = VectorCellFunction(state).fill([1.0, 0.0, 0.0])

    Ms_val = 8e5
    Ms_data = np.zeros((nx, 2, 2))
    Ms_data[:d1, :, :] = Ms_val
    state.material.Ms = CellFunction(state, tensor=state.tensor(Ms_data))
    DemagField().register(state)

    h = be.to_numpy(state.h_demag.tensor)
    h_film = -Ms_val * d0 / (d1 + d0)
    h_gap = Ms_val * d1 / (d1 + d0)

    assert h[:d1, :, :, 0].mean() == pytest.approx(h_film, rel=1e-6)
    assert h[d1:, :, :, 0].mean() == pytest.approx(h_gap, rel=1e-6)
    assert h[:, :, :, 1].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[:, :, :, 2].mean() == pytest.approx(0.0, abs=1e-6)


def test_demag_node_pbc_uniform():
    """For uniform FEM magnetization with full PBC, h_demag should be zero."""
    mesh = Mesh(
        (5, 5, 5),
        (1e-9, 1e-9, 1e-9),
        pbc=True,
    )
    state = State(mesh)
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)

    h = be.to_numpy(state.h_demag.tensor)
    assert h[..., 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 1].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 2].mean() == pytest.approx(0.0, abs=1e-6)


def test_demag_pseudo_pbc_cube():
    """Pseudo PBC single-cell cube: Nx = Ny = Nz = 1/3."""
    state = State(Mesh((1, 1, 1), (1e-9, 1e-9, 1e-9), pbc=(0, 0, 0)))
    state.m = VectorCellFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor[0, 0, 0].real)
    np.testing.assert_allclose(h, [-1.0 / 3.0, 0.0, 0.0], atol=1e-6)


def test_demag_pseudo_pbc_cylinder():
    """Pseudo PBC long cylinder: Nx -> 0, Ny = Nz -> 1/2."""
    state = State(Mesh((1, 1, 1), (1e-9, 1e-9, 1e-9), pbc=(10, 0, 0)))
    state.m = VectorCellFunction(state).fill([0.0, 1.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor[0, 0, 0].real)
    np.testing.assert_allclose(h, [0.0, -0.5, 0.0], atol=1e-2)


def test_demag_pseudo_pbc_film():
    """Pseudo PBC thin film: Nx = Ny -> 0, Nz -> 1."""
    state = State(Mesh((1, 1, 1), (1e-9, 1e-9, 1e-9), pbc=(10, 10, 0)))
    state.m = VectorCellFunction(state).fill([0.0, 0.0, 1.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor[0, 0, 0].real)
    np.testing.assert_allclose(h, [0.0, 0.0, -1.0], atol=5e-2)
