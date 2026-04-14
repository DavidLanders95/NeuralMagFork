# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from neuralmag import *

be = config.backend


def test_h():
    """3D open-BC cube: average demagnetization factor N = 1/3."""
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    assert be.to_numpy((state.h_demag.tensor * state.m.tensor).sum() / 6**3) == pytest.approx(-1 / 3)


def test_h_single_cell():
    """Single-cell open-BC cube: Nx = Ny = Nz = 1/3."""
    dx = (1e-9, 1e-9, 1e-9)
    for m_dir, expected in [
        ([1, 0, 0], [-1 / 3, 0, 0]),
        ([0, 1, 0], [0, -1 / 3, 0]),
        ([0, 0, 1], [0, 0, -1 / 3]),
    ]:
        state = State(Mesh((1, 1, 1), dx))
        state.m = VectorCellFunction(state).fill(m_dir)
        state.material.Ms = CellFunction(state).fill(1.0)
        DemagField().register(state)
        h = be.to_numpy(state.h_demag.tensor[0, 0, 0])
        np.testing.assert_allclose(h, expected, atol=1e-6)


def test_h_2d():
    """2D open-BC: approximate demagnetization factor."""
    mesh = Mesh((5, 5), (1e-9, 1e-9, 5e-9))
    state = State(mesh)
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    assert be.to_numpy((state.h_demag.tensor * state.m.tensor).sum() / 6**2) == pytest.approx(-1 / 3, rel=1e-1)


def test_E():
    """Demag energy for unit-Ms cube."""
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    assert be.to_numpy(state.E_demag) == pytest.approx(2.6179938794166683e-32)


def test_rename():
    """Demag field with custom name suffix."""
    mesh = Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9))
    state = State(mesh)
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state, "d")
    assert be.to_numpy((state.h_d.tensor * state.m.tensor).sum() / 6**3) == pytest.approx(-1 / 3)
    assert be.to_numpy(state.E_d) == pytest.approx(2.6179938794166683e-32)


def test_h_cell_pbc_uniform():
    """Uniform magnetization with full PBC: h_demag should be zero."""
    state = State(Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9), pbc=True))
    state.m = VectorCellFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor)
    assert h[..., 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 1].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 2].mean() == pytest.approx(0.0, abs=1e-6)


def test_h_cell_pbc_stripes():
    """Periodic stripes from Bruckner et al., Sci. Rep. 11, 9202 (2021).

    Infinite periodic thin films (thickness d1) separated by non-magnetic gaps
    (thickness d0), magnetized perpendicular to the film plane. The analytical
    demag field is piecewise constant:
        h_x(film) = -Ms * d0 / (d1 + d0)
        h_x(gap)  =  Ms * d1 / (d1 + d0)
    """
    d1, d0 = 4, 6
    nx = d1 + d0
    state = State(Mesh((nx, 2, 2), (1e-9, 1e-9, 1e-9), pbc=True))
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


def _test_pseudo_pbc_geometry(pbc, cases, atol):
    """Test pseudo PBC demag field for all magnetization directions."""
    n, dx = (1, 1, 1), (1e-9, 1e-9, 1e-9)
    for m_dir, expected in cases:
        state = State(Mesh(n, dx, pbc=pbc))
        state.m = VectorCellFunction(state).fill(m_dir)
        state.material.Ms = CellFunction(state).fill(1.0)
        DemagField().register(state)
        h = be.to_numpy(state.h_demag.tensor[0, 0, 0].real)
        np.testing.assert_allclose(h, expected, atol=atol)


def test_h_cell_pseudo_pbc_cube():
    """Pseudo PBC cube: Nx = Ny = Nz = 1/3."""
    _test_pseudo_pbc_geometry(
        (0, 0, 0),
        [
            ([1, 0, 0], [-1 / 3, 0, 0]),
            ([0, 1, 0], [0, -1 / 3, 0]),
            ([0, 0, 1], [0, 0, -1 / 3]),
        ],
        atol=1e-6,
    )


def test_h_cell_pseudo_pbc_cylinder():
    """Pseudo PBC long cylinder along x: Nx -> 0, Ny = Nz -> 1/2."""
    _test_pseudo_pbc_geometry(
        (10, 0, 0),
        [([1, 0, 0], [0, 0, 0]), ([0, 1, 0], [0, -0.5, 0]), ([0, 0, 1], [0, 0, -0.5])],
        atol=1e-2,
    )


def test_h_cell_pseudo_pbc_film():
    """Pseudo PBC thin film in xy-plane: Nx = Ny -> 0, Nz -> 1."""
    _test_pseudo_pbc_geometry(
        (10, 10, 0),
        [([1, 0, 0], [0, 0, 0]), ([0, 1, 0], [0, 0, 0]), ([0, 0, 1], [0, 0, -1])],
        atol=5e-2,
    )


def test_h_fem_pbc_uniform():
    """Uniform FEM magnetization with full 3D PBC: h_demag should be zero."""
    state = State(Mesh((5, 5, 5), (1e-9, 1e-9, 1e-9), pbc=True))
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    state.material.Ms = CellFunction(state).fill(1.0)
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor)
    assert h[..., 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 1].mean() == pytest.approx(0.0, abs=1e-6)
    assert h[..., 2].mean() == pytest.approx(0.0, abs=1e-6)


def test_h_fem_pbc_stripes():
    """FEM PBC stripes: analytical solution from Bruckner et al.

    Interior film nodes (away from Ms discontinuity) should match the
    analytical field. Nodes in the non-magnetic gap have zero lumped mass.
    """
    d1, d0 = 4, 6
    nx = d1 + d0
    state = State(Mesh((nx, 2, 2), (1e-9, 1e-9, 1e-9), pbc=True))
    state.m = VectorFunction(state).fill([1.0, 0.0, 0.0])
    Ms_val = 8e5
    Ms_data = np.zeros((nx, 2, 2))
    Ms_data[:d1, :, :] = Ms_val
    state.material.Ms = CellFunction(state, tensor=state.tensor(Ms_data))
    DemagField().register(state)
    h = be.to_numpy(state.h_demag.tensor)

    h_film_expected = -Ms_val * d0 / (d1 + d0)
    h_interior = h[1 : d1 - 1, :, :, :]
    assert h_interior[..., 0].mean() == pytest.approx(h_film_expected, rel=5e-2)
    assert h_interior[..., 1].mean() == pytest.approx(0.0, abs=1e-3)
    assert h_interior[..., 2].mean() == pytest.approx(0.0, abs=1e-3)
