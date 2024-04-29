import copy
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import bisect

from neuralmag import *

mu0 = 4 * np.pi * 1e-7

@pytest.mark.sp
def test_sp3():
    mu0 = 4 * np.pi * 1e-7
    N = 16  # discretisation in one dimension
    cubesize = 100e-9  # cube edge length (m)
    cellsize = cubesize / N  # discretisation in all three dimensions.
    init_m = "vortex"

    mesh = Mesh([N, N, N], [cellsize, cellsize, cellsize], [0.0, 0.0, 0.0])
    state = State(mesh)

    ExchangeField().register(state, "exchange")
    UniaxialAnisotropyField().register(state, "aniso")
    DemagField().register(state, "demag")
    TotalField("exchange", "demag", "aniso").register(state)

    L_array = np.linspace(8.3, 8.6, 5)
    flower_energies, vortex_energies = [], []
    for L in L_array:

        lex = cubesize / L  # exchange length.
        Km = 1e6  # magnetostatic energy density (J/m**3)
        Ms = np.sqrt(2 * Km / mu0)  # magnetisation saturation (A/m)
        A = 0.5 * mu0 * Ms**2 * lex**2  # exchange energy constant
        K = 0.1 * Km  # Uniaxial anisotropy constant

        state.material.Ms = Ms
        state.material.A = A
        state.material.alpha = 0.5
        state.material.Ku = 0.1 * Km
        state.material.Ku_axis = [0, 0, 1]

        # Flower state
        state.m = VectorFunction(state).fill((0.0, 0.0, 1))
        llg = LLGSolver(state)
        llg.step(20e-10)
        # E_flower = state.E
        flower_energies.append(state.E)

        # Vortex state
        x, y, z = state.coordinates(spaces=("n", "n", "n"))
        out = np.stack([np.ones_like(x) * 5e-9, z - 50e-9, 50e-9 - y], axis=3)
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        state.m = VectorFunction(state, tensor=state.tensor(out / norm))
        llg = LLGSolver(state)
        llg.step(20e-10)
        # E_vortex = state.E
        vortex_energies.append(state.E)

    delta_E = np.subtract(vortex_energies, flower_energies)
    # Find indices where there's a sign change in mz
    sign_change_indices = np.where(np.diff(np.sign(delta_E)))[0]

    # For each sign change, interpolate to find the zero crossing
    zero_crossings = []
    for index in sign_change_indices:
        # Get the two points around the sign change
        x1, x2 = L_array[index], L_array[index + 1]
        y1, y2 = delta_E[index], delta_E[index + 1]

        # Perform linear interpolation
        zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
        zero_crossings.append(zero_crossing)

    cross = abs(zero_crossings[0])

    try:
        assert 8.41 < cross < 8.51
    except AssertionError:
        plt.plot(L_array, flower_energies, label="Flower energies")
        plt.plot(L_array, vortex_energies, label="Vortex energies")
        plt.legend()
        plt.xlabel("L (lex)")
        plt.ylabel("E (J)")
        plt.tight_layout()
        plt.savefig("tests/test_artifacts/sp3_failure_plot.png")
        raise
