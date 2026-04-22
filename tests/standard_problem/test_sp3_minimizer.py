import matplotlib.pyplot as plt
import numpy as np
import pytest

from neuralmag import *

mu0 = 4 * np.pi * 1e-7


@pytest.mark.slow
def test_sp3_minimizer():
    N = 16
    cubesize = 100e-9
    cellsize = cubesize / N

    mesh = Mesh([N, N, N], [cellsize, cellsize, cellsize], [0.0, 0.0, 0.0])
    state = State(mesh)
    state.m = VectorFunction(state).fill((0.0, 0.0, 1.0))

    ExchangeField().register(state, "exchange")
    UniaxialAnisotropyField().register(state, "aniso")
    DemagField().register(state, "demag")
    TotalField("exchange", "demag", "aniso").register(state)

    L_array = np.linspace(8.3, 8.6, 5)
    flower_energies, vortex_energies = [], []
    for L in L_array:
        lex = cubesize / L
        Km = 1e6
        Ms = np.sqrt(2 * Km / mu0)
        A = 0.5 * mu0 * Ms**2 * lex**2

        state.material.Ms = Ms
        state.material.A = A
        state.material.alpha = 0.5
        state.material.Ku = 0.1 * Km
        state.material.Ku_axis = [0, 0, 1]

        state.m = VectorFunction(state).fill((0.0, 0.0, 1.0))
        minimizer = EnergyMinimizer(state, tol=1e3, max_iter=2000)
        minimizer.minimize()
        flower_energies.append(state.E)

        x, y, z = state.coordinates(spaces=("n", "n", "n"))
        out = np.stack([np.ones_like(x) * 5e-9, z - 50e-9, 50e-9 - y], axis=3)
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        state.m = VectorFunction(state, tensor=state.tensor(out / norm))
        minimizer = EnergyMinimizer(state, tol=1e3, max_iter=2000)
        minimizer.minimize()
        vortex_energies.append(state.E)

    delta_E = np.subtract(vortex_energies, flower_energies)
    sign_change_indices = np.where(np.diff(np.sign(delta_E)))[0]

    zero_crossings = []
    for index in sign_change_indices:
        x1, x2 = L_array[index], L_array[index + 1]
        y1, y2 = delta_E[index], delta_E[index + 1]
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
        plt.savefig("tests/test_artifacts/sp3_minimizer_failure_plot.png")
        raise