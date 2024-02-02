import numpy as np
import pytest
import torch

from neuralmag import *


def test_rw_function(state, tmp_path):
    f = Function(state).from_constant(2.0)
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.ftype == "node"
    assert g.state.mesh.dim == 3
    assert g.avg().cpu() == pytest.approx(2.0)


def test_rw_vector_function(state, tmp_path):
    f = VectorFunction(state).from_constant((1.0, 2.0, 3.0))
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.ftype == "node"
    assert g.state.mesh.dim == 3
    assert g.avg().cpu() == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function(state, tmp_path):
    f = CellFunction(state).from_constant(2.0)
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.ftype == "cell"
    assert g.state.mesh.dim == 3
    assert g.avg().cpu() == pytest.approx(2.0)


def test_rw_vector_cell_function(state, tmp_path):
    f = VectorCellFunction(state).from_constant((1.0, 2.0, 3.0))
    state.write_vti(f, tmp_path / "f.vti")
    g = state.read_vti(tmp_path / "f.vti")
    assert g.ftype == "cell"
    assert g.state.mesh.dim == 3
    assert g.avg().cpu() == pytest.approx((1.0, 2.0, 3.0))


def test_rw_function_2d(state2d, tmp_path):
    f = Function(state2d).from_constant(2.0)
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.ftype == "node"
    assert g.state.mesh.dim == 2
    assert g.avg().cpu() == pytest.approx(2.0)


def test_rw_vector_function_2d(state2d, tmp_path):
    f = VectorFunction(state2d).from_constant((1.0, 2.0, 3.0))
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.ftype == "node"
    assert g.state.mesh.dim == 2
    assert g.avg().cpu() == pytest.approx((1.0, 2.0, 3.0))


def test_rw_cell_function_2d(state2d, tmp_path):
    f = CellFunction(state2d).from_constant(2.0)
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.ftype == "cell"
    assert g.state.mesh.dim == 2
    assert g.avg().cpu() == pytest.approx(2.0)


def test_rw_vector_cell_function_2d(state2d, tmp_path):
    f = VectorCellFunction(state2d).from_constant((1.0, 2.0, 3.0))
    state2d.write_vti(f, tmp_path / "f.vti")
    g = state2d.read_vti(tmp_path / "f.vti")
    assert g.ftype == "cell"
    assert g.state.mesh.dim == 2
    assert g.avg().cpu() == pytest.approx((1.0, 2.0, 3.0))
