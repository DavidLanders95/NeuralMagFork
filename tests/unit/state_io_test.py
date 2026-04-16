import pytest

from neuralmag import *

be = config.backend


def test_rw_function(state, tmp_path):
    f = Function(state).fill(2.0)
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nnn"
    assert g.state.mesh.dim == 3
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_function(state, tmp_path):
    f = VectorFunction(state).fill((1.0, 2.0, 3.0))
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nnn"
    assert g.state.mesh.dim == 3
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function(state, tmp_path):
    f = CellFunction(state).fill(2.0)
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.spaces == "ccc"
    assert g.state.mesh.dim == 3
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_cell_function(state, tmp_path):
    f = VectorCellFunction(state).fill((1.0, 2.0, 3.0))
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.spaces == "ccc"
    assert g.state.mesh.dim == 3
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_function_2d(state2d, tmp_path):
    f = Function(state2d).fill(2.0)
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nn"
    assert g.state.mesh.dim == 2
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_function_2d(state2d, tmp_path):
    f = VectorFunction(state2d).fill((1.0, 2.0, 3.0))
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nn"
    assert g.state.mesh.dim == 2
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function_2d(state2d, tmp_path):
    f = CellFunction(state2d).fill(2.0)
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "cc"
    assert g.state.mesh.dim == 2
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_cell_function_2d(state2d, tmp_path):
    f = VectorCellFunction(state2d).fill((1.0, 2.0, 3.0))
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "cc"
    assert g.state.mesh.dim == 2
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_function_1d(state1d, tmp_path):
    f = Function(state1d).fill(2.0)
    state1d.write_vti(f, tmp_path / "f.vti")
    g = state1d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "n"
    assert g.state.mesh.dim == 1
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_function_1d(state1d, tmp_path):
    f = VectorFunction(state1d).fill((1.0, 2.0, 3.0))
    state1d.write_vti(f, tmp_path / "f.vti")
    g = state1d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "n"
    assert g.state.mesh.dim == 1
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function_1d(state1d, tmp_path):
    f = CellFunction(state1d).fill(2.0)
    state1d.write_vti(f, tmp_path / "f.vti")
    g = state1d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "c"
    assert g.state.mesh.dim == 1
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_cell_function_1d(state1d, tmp_path):
    f = VectorCellFunction(state1d).fill((1.0, 2.0, 3.0))
    state1d.write_vti(f, tmp_path / "f.vti")
    g = state1d.read_vti(tmp_path / "f.vti")
    assert g.spaces == "c"
    assert g.state.mesh.dim == 1
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_function_pbc(state_pbc, tmp_path):
    f = Function(state_pbc).fill(2.0)
    state_pbc.write_vti(f, tmp_path / "f.vti")
    g = state_pbc.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nnn"
    assert g.tensor_shape == f.tensor_shape
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_function_pbc(state_pbc, tmp_path):
    f = VectorFunction(state_pbc).fill((1.0, 2.0, 3.0))
    state_pbc.write_vti(f, tmp_path / "f.vti")
    g = state_pbc.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nnn"
    assert g.tensor_shape == f.tensor_shape
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function_pbc(state_pbc, tmp_path):
    f = CellFunction(state_pbc).fill(2.0)
    state_pbc.write_vti(f, tmp_path / "f.vti")
    g = state_pbc.read_vti(tmp_path / "f.vti")
    assert g.spaces == "ccc"
    assert be.to_numpy(g.avg()) == pytest.approx(2.0)


def test_rw_vector_function_pbc_mixed(state_pbc_mixed, tmp_path):
    f = VectorFunction(state_pbc_mixed).fill((1.0, 2.0, 3.0))
    state_pbc_mixed.write_vti(f, tmp_path / "f.vti")
    g = state_pbc_mixed.read_vti(tmp_path / "f.vti")
    assert g.spaces == "nnn"
    assert g.tensor_shape == f.tensor_shape
    assert be.to_numpy(g.avg()) == pytest.approx((1.0, 2.0, 3.0))
