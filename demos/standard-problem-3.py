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
from scipy import constants

import neuralmag as nm

pv.set_jupyter_backend("static")


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


def minimise_system_energy(L, m_init):
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

    # relax the system
    llg = nm.LLGSolver(state)
    llg.relax()
    return state


# ### Relaxed magnetization states
#
# **Vortex** state:

# Minimize energy and write magnetization to file
state = minimise_system_energy(8, m_init_vortex)
state.write_vti(["m"], "standard-problem-3/vortex.vti")

# Plot the vortex state
mesh = pv.read("standard-problem-3/vortex.vti")
glyphs = mesh.glyph(orient="m", scale="m", factor=1e-8)
p = pv.Plotter()
p.add_mesh(glyphs, color="white", lighting=True, smooth_shading=True)
p.show()


# **Flower** state:

# Minimize energy and write magnetization to file
state = minimise_system_energy(8, m_init_flower)
state.write_vti(["m"], "standard-problem-3/flower.vti")

# Plot the flower state
mesh = pv.read("standard-problem-3/flower.vti")
glyphs = mesh.glyph(orient="m", scale="m", factor=1e-8)
p = pv.Plotter()
p.add_mesh(glyphs, color="white", lighting=True, smooth_shading=True)
p.show()


# ### Energy crossing
#
# We can plot the energies of both vortex and flower states as a function of cube edge length $L$. This will give us an idea where the state transition occurrs. We can achieve that by simply looping over the edge lengths $L$ of interest, computing the energy of both vortex and flower states, and finally, plotting the energy dependence.

vortex_energies, flower_energies = [], []
L_array = [8.3, 8.4, 8.5]
for L in L_array:
    vortex = minimise_system_energy(L, m_init_vortex)
    flower = minimise_system_energy(L, m_init_flower)
    vortex_energies.append(nm.config.backend.to_numpy(vortex.E))
    flower_energies.append(nm.config.backend.to_numpy(flower.E))

plt.figure(figsize=(8, 4))
plt.plot(L_array, vortex_energies, "o-", label="vortex")
plt.plot(L_array, flower_energies, "o-", label="flower")
plt.xlabel("L (lex)")
plt.ylabel("E (J)")
plt.grid()
plt.legend()
plt.show()


# From the plot, we can see that the energy crossing occurrs between $8.3l_\text{ex}$ and $8.5l_\text{ex}$.

# ## References
#
# [1] µMAG Site Directory http://www.ctcms.nist.gov/~rdm/mumag.org.
#
# This tutorial was adapted from [Ubermag](https://ubermag.github.io/).
