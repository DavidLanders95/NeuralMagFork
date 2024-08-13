import numpy as np
import pytest
import torch

from neuralmag import *


def test_volume():
    mesh = Mesh((10, 10, 1), (1.0, 2.0, 3.0))
    assert mesh.volume == pytest.approx(600.0)


def test_volume_2D():
    mesh = Mesh((10, 10), (1.0, 2.0, 3.0))
    assert mesh.volume == pytest.approx(600.0)


def test_num_cells():
    mesh = Mesh((10, 10, 1), (1.0, 2.0, 3.0))
    assert mesh.num_cells == 100


def test_num_cells_2D():
    mesh = Mesh((10, 10), (1.0, 2.0, 3.0))
    assert mesh.num_cells == 100


def test_str():
    mesh = Mesh((10, 10, 10), (1e-9, 2e-9, 3e-9))
    assert str(mesh) == "10x10x10_1e-09x2e-09x3e-09"


def test_str_1D():
    mesh = Mesh((10,), (1e-9, 2e-9, 3e-9))
    assert str(mesh) == "10_1e-09x2e-09x3e-09"
