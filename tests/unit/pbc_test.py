# SPDX-License-Identifier: MIT

import numpy as np

from neuralmag import *

be = config.backend


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
