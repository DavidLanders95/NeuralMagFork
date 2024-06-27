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
