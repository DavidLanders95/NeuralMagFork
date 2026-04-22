#!/usr/bin/env python
# coding: utf-8

# # Standard problem 3
#
# ## Problem specification
#
# This problem is to calculate a single domain limit of a cubic magnetic particle. This is the size $L$ of equal energy for the so-called flower state (which one may also call a splayed state or a modified single-domain state) on the one hand, and the vortex or curling state on the other hand.
#
# Geometry:
#
# A cube with edge length, $L$, expressed in units of the intrinsic length scale, $l_\text{ex} = \sqrt{A/K_\text{m}}$, where $K_\text{m}$ is a magnetostatic energy density, $K_\text{m} = \frac{1}{2}\mu_{0}M_\text{s}^{2}$.
#
# Material parameters:
#
# - uniaxial anisotropy $K_\text{u}$ with $K_\text{u} = 0.1 K_\text{m}$, and with the easy axis directed parallel to a principal axis of the cube (0, 0, 1),
# - exchange energy constant is $A = \frac{1}{2}\mu_{0}M_\text{s}^{2}l_\text{ex}^{2}$.
#
# More details about the standard problem 3 can be found in Ref. 1.
#
# ## Simulation
#
# Firstly, we import all necessary modules and configure pyvista for static rendering.

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import time
from scipy import constants

import neuralmag as nm

pv.set_jupyter_backend("static")

SOLVER_LABELS = {
    "llg": "LLG relax",
    "bb": "BB steepest descent",
}

# Switch between the damped-LLG relaxer and the Barzilai-Borwein steepest descent driver.
SELECTED_METHOD = "bb"


# The following two functions are used for initialising the system's magnetisation [1].


# Function for initiaising the flower state.
def m_init_flower(state):
    return nm.VectorFunction(state).fill((0.0, 0.0, 1))


# Function for initialising the vortex state.
def m_init_vortex(state):
    x, y, z = state.coordinates(spaces="nnn", numpy=True)
    m = np.stack([np.ones_like(x) * 1e-9, z, -y], axis=-1)
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    return nm.VectorFunction(state, tensor=state.tensor(m / norm))


# The following function is used for convenience. It takes two arguments:
#
# - $L$ - the cube edge length in units of $l_\text{ex}$, and
# - the function for initialising the system's magnetization.
#
# It returns the relaxed system object.
#
# Please refer to other tutorials for more details on how to create system objects and drive them using specific drivers.


def minimise_system_energy(L, m_init, method=SELECTED_METHOD):
    N = 10  # discretisation in one dimension
    Ms = 8e5  # saturation magnetization
    A = 13e-12  # exchange constant
    Km = constants.mu_0 * Ms**2 / 2.0  # effective anisotropy
    lex = (A / Km) ** 0.5  # exchange length
    dx = L * lex / N

    mesh = nm.Mesh((N, N, N), (dx, dx, dx), origin=(-L * lex / 2, -L * lex / 2, -L * lex / 2))
    state = nm.State(mesh)

    state.material.Ms = Ms
    state.material.A = A
    state.material.alpha = 0.5
    state.material.Ku = 0.1 * Km
    state.material.Ku_axis = [0, 0, 1]

    state.m = m_init(state)

    nm.ExchangeField().register(state, "exchange")
    nm.UniaxialAnisotropyField().register(state, "aniso")
    nm.DemagField().register(state, "demag")
    nm.TotalField("exchange", "demag", "aniso").register(state)

    # Both methods aim at the same static state, but the BB driver is not a time integrator.
    t_start = time.perf_counter()
    if method == "llg":
        solver = nm.LLGSolver(state)
        solver.relax()
        meta = {"runtime_s": time.perf_counter() - t_start, "iterations": None}
    elif method == "bb":
        solver = nm.EnergyMinimizer(state, tol=1e3, max_iter=2000)
        solver.minimize()
        meta = {"runtime_s": time.perf_counter() - t_start, "iterations": solver.n_iter}
    else:
        raise ValueError(f"Unknown minimization method: {method}")

    return state, meta


def estimate_crossing(L_array, flower_energies, vortex_energies):
    delta_E = np.subtract(vortex_energies, flower_energies)
    sign_change_indices = np.where(np.diff(np.sign(delta_E)))[0]
    if len(sign_change_indices) == 0:
        return None, delta_E

    zero_crossings = []
    for index in sign_change_indices:
        x1, x2 = L_array[index], L_array[index + 1]
        y1, y2 = delta_E[index], delta_E[index + 1]
        zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
        zero_crossings.append(zero_crossing)

    return abs(zero_crossings[0]), delta_E


def compare_minimizers(L_array, methods=("llg", "bb")):
    comparison = {}
    for method in methods:
        vortex_energies, flower_energies = [], []
        runtimes = []
        iterations = []

        for L in L_array:
            vortex, vortex_meta = minimise_system_energy(L, m_init_vortex, method=method)
            flower, flower_meta = minimise_system_energy(L, m_init_flower, method=method)

            vortex_energies.append(float(np.asarray(nm.config.backend.to_numpy(vortex.E))))
            flower_energies.append(float(np.asarray(nm.config.backend.to_numpy(flower.E))))
            runtimes.append(vortex_meta["runtime_s"] + flower_meta["runtime_s"])

            if vortex_meta["iterations"] is not None:
                iterations.append(vortex_meta["iterations"] + flower_meta["iterations"])

        crossing, delta_E = estimate_crossing(L_array, flower_energies, vortex_energies)
        comparison[method] = {
            "vortex_energies": vortex_energies,
            "flower_energies": flower_energies,
            "delta_E": delta_E,
            "crossing": crossing,
            "total_runtime_s": float(np.sum(runtimes)),
            "avg_runtime_s": float(np.mean(runtimes)),
            "avg_iterations": float(np.mean(iterations)) if iterations else None,
        }

    return comparison


def write_and_plot_state(state, filename):
    state.write_vti(["m"], filename)
    mesh = pv.read(filename)
    glyphs = mesh.glyph(orient="m", scale="m", factor=1e-8)
    p = pv.Plotter()
    p.add_mesh(glyphs, color="white", lighting=True, smooth_shading=True)
    p.show()


def plot_selected_method_energies(L_array, comparison, method):
    method_data = comparison[method]
    plt.figure(figsize=(8, 4))
    plt.plot(L_array, method_data["vortex_energies"], "o-", label="vortex")
    plt.plot(L_array, method_data["flower_energies"], "o-", label="flower")
    plt.xlabel("L (lex)")
    plt.ylabel("E (J)")
    plt.title(f"SP3 energies with {SOLVER_LABELS[method]}")
    plt.grid()
    plt.legend()
    plt.show()


def plot_method_comparison(L_array, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, values in comparison.items():
        label = SOLVER_LABELS[method]
        axes[0].plot(L_array, values["flower_energies"], "o--", label=f"flower ({label})")
        axes[0].plot(L_array, values["vortex_energies"], "o-", label=f"vortex ({label})")

        crossing_label = label
        if values["crossing"] is not None:
            crossing_label += f", L*={values['crossing']:.3f}"
        axes[1].plot(L_array, values["delta_E"], "o-", label=crossing_label)

    axes[0].set_xlabel("L (lex)")
    axes[0].set_ylabel("E (J)")
    axes[0].set_title("Method comparison by state")
    axes[0].grid()
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    axes[1].set_xlabel("L (lex)")
    axes[1].set_ylabel(r"$E_\mathrm{vortex} - E_\mathrm{flower}$ (J)")
    axes[1].set_title("Crossing comparison")
    axes[1].grid()
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def print_method_summary(comparison):
    print("Method comparison for standard problem 3")
    for method, values in comparison.items():
        line = (
            f"- {SOLVER_LABELS[method]}: crossing ~= {values['crossing']:.3f}, "
            f"total runtime = {values['total_runtime_s']:.2f}s, "
            f"average runtime per L = {values['avg_runtime_s']:.2f}s"
        )
        if values["avg_iterations"] is not None:
            line += f", average BB iterations per L = {values['avg_iterations']:.1f}"
        print(line)


def main():
    method = SELECTED_METHOD

    # ### Relaxed magnetization states
    #
    # **Vortex** state:

    # Minimize energy and write magnetization to file.
    state, _ = minimise_system_energy(8, m_init_vortex, method=method)
    write_and_plot_state(state, f"standard-problem-3/vortex-{method}.vti")

    # **Flower** state:

    # Minimize energy and write magnetization to file.
    state, _ = minimise_system_energy(8, m_init_flower, method=method)
    write_and_plot_state(state, f"standard-problem-3/flower-{method}.vti")

    # ### Energy crossing
    #
    # We can plot the energies of both vortex and flower states as a function of cube edge length $L$.
    # This gives an estimate of the state transition region. We also compare the two minimization
    # methods directly by running the same SP3 sweep with both drivers.

    L_array = [8.3, 8.4, 8.5]
    comparison = compare_minimizers(L_array)

    plot_selected_method_energies(L_array, comparison, method)
    plot_method_comparison(L_array, comparison)
    print_method_summary(comparison)


if __name__ == "__main__":
    main()


# From the plot, we can see that the energy crossing occurrs between $8.3l_\text{ex}$ and $8.5l_\text{ex}$.

# ## References
#
# [1] µMAG Site Directory http://www.ctcms.nist.gov/~rdm/mumag.org.
#
# This tutorial was adapted from [Ubermag](https://ubermag.github.io/).
