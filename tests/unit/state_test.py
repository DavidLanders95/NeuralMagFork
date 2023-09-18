import numpy as np
import pytest
import torch

from nmagnum import *


def test_asdf():
    mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
    c = State(mesh)

    c.a = 1
    c.b = lambda a: 2*a
    c.c = lambda b: 2*b
    c.material.d = lambda a: 2*a
    assert c.a == 1
    assert c.b == 2
    assert c.c == 4

    assert c.material.d == 2
