#!/usr/bin/env python
# coding: utf-8

# # Adding DMI via Lifshitz Invariants
# This notebook shows how to implement **Dzyaloshinskii–Moriya Interaction (DMI)** energy terms using **Lifshitz invariants**.
#
# ## What are Lifshitz Invariants?
#
# The **Dzyaloshinskii–Moriya interaction (DMI)** can be written compactly using **Lifshitz invariants (LI)**.
#
# A Lifshitz invariant is an antisymmetric combination of magnetization components and their derivatives:
#
# $$\mathcal{L}_{ij}^{k} = m_i \partial_k m_j - m_j \partial_k m_i, \quad i,j,k \in \{x,y,z\}.$$
#
#

# ## Adding LI as an energy contribution
# As an example lets take a system with exchange and a single LI of type $\mathcal{L}_{xz}^x$.
#
# Each LI corresponds to a possible DMI contribution.
# For example:
#
# $$
# w = D_{xzx} \, \mathcal{L}_{xz}^{x} = D_{xzx} \,( m_x \,\partial_x m_z - m_z \,\partial_x m_x ).
# $$
#
# In this system we would expect the ground state to be a cycloid propagating in the x direction.
# ### Setup
# First we import neuralmag and create a 2D mesh.

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import neuralmag as nm
from _static_method_compare import SOLVER_LABELS, compare_static_methods, print_static_method_summary

nm.config.dtype = "float64"
pv.set_jupyter_backend("static")

METHODS = ("llg", "bb")

def plot_li_comparison(comparison, output_prefix):
	plotter = pv.Plotter(shape=(1, len(METHODS)))
	for column, method in enumerate(METHODS):
		comparison[method]["state"].write_vti(["m"], f"{output_prefix}-{method}.vti")
		grid = pv.read(f"{output_prefix}-{method}.vti")
		grid["m_z"] = grid["m"][:, 2]
		glyphs = grid.glyph(orient="m", scale="m", factor=2e-9)

		plotter.subplot(0, column)
		plotter.add_text(SOLVER_LABELS[method], font_size=12)
		plotter.add_mesh(glyphs, scalars="m_z", lighting=True, smooth_shading=True)
		plotter.show_axes()

	plotter.link_views()
	plotter.show()


# ### Defining Energy Terms
#
# We set the material parameters as usual.
#
# To add a Lifshitz invariant, specify the indices $i,j,k$ as a string (e.g., "xzx") to `nm.LIField`, and set the corresponding DMI constant `Dijk` in the material.

def build_xzx_state():
	mesh = nm.Mesh((20, 20), (1e-9, 1e-9, 1e-9))
	state = nm.State(mesh)

	state.m = nm.VectorFunction(state).fill((0, 0, 1))
	state.material.Ms = 0.86e6
	state.material.alpha = 1
	state.material.A = 1.3e-11
	nm.ExchangeField().register(state, "exchange")

	state.material.Dxzx = 15e-3
	nm.LIField("xzx").register(state, "li_xzx")

	nm.TotalField("exchange", "li_xzx").register(state)
	return state


comparison_xzx = compare_static_methods(build_xzx_state, methods=METHODS, llg_runner=lambda solver: solver.relax(1e9))
print_static_method_summary("LI xzx example", comparison_xzx)


# ### Visualization
plot_li_comparison(comparison_xzx, "LI/m-xzx")


# ## Another Example: Different Invariant
#
# Let's try a different Lifshitz invariant, "xzy". This should give us a helix propagating the the $y$ direction.

def build_xzy_state():
	mesh = nm.Mesh((20, 20), (1e-9, 1e-9, 1e-9))
	state = nm.State(mesh)

	state.m = nm.VectorFunction(state).fill((0, 0, 1))
	state.material.Ms = 0.86e6
	state.material.alpha = 1
	state.material.A = 1.3e-11
	nm.ExchangeField().register(state, "exchange")

	state.material.Dxzy = 15e-3
	nm.LIField("xzy").register(state, "li_xzy")

	nm.TotalField("exchange", "li_xzy").register(state)
	return state


comparison_xzy = compare_static_methods(build_xzy_state, methods=METHODS, llg_runner=lambda solver: solver.relax(1e9))
print_static_method_summary("LI xzy example", comparison_xzy)
plot_li_comparison(comparison_xzy, "LI/m-xzy")


# ## Using this for particular crystallographic groups
#
# In NeuralMag we have already created energy terms for the two most common DMI classes:
#
# - **Bulk (T symmetry)**:
#   - $w_{\text{bulk}} = D \, \mathbf{m} \cdot (\nabla \times \mathbf{m})$
#   - Equivalent to a sum of Lifshitz invariants: $D \, (\mathcal{L}_{yz}^{(x)} + \mathcal{L}_{zx}^{(y)} + \mathcal{L}_{xy}^{(z)})$.
#
# - **Interfacial ($C_{\infty v}$, "Néel" type, thin film)**:
#   - $w_{\text{int}} = D \, [m_z \, \nabla \cdot \mathbf{m} - (\mathbf{m} \cdot \nabla) m_z]$
#   - Equivalent to a sum of Lifshitz invariants:
#   $D \, (\mathcal{L}_{zx}^{(x)} + \mathcal{L}_{zy}^{(y)})$.
#
# The table below lists the DMI energy densities for various crystallographic point groups:
#
# | Point Group | DMI Energy Density |
# |-------------|---------------------|
# | O, T, $D_n \, (n \geq 3)$ | $D \, (\mathcal{L}^{x}_{yz} + \mathcal{L}^{y}_{zx})$ |
# | $C_{nv} \, (n \geq 3)$ | $D \, (\mathcal{L}^{x}_{zx} - \mathcal{L}^{y}_{yz})$ |
# | $D_{2d}$ | $D \, (\mathcal{L}^{x}_{yz} - \mathcal{L}^{y}_{zx})$ |
# | $S_{4}$ | $D_{0} \, (\mathcal{L}^{x}_{zx} + \mathcal{L}^{y}_{yz}) + D_{1} \, (\mathcal{L}^{x}_{yz} - \mathcal{L}^{y}_{zx})$ |
# | $C_{n} \, (n \geq 3)$ | $D_{0} \, (\mathcal{L}^{x}_{zx} - \mathcal{L}^{y}_{yz}) + D_{1} \, (\mathcal{L}^{x}_{yz} + \mathcal{L}^{y}_{zx})$ |
# | $D_{2}$, $C_{2v}$ | $D_{0} \, \mathcal{L}^{x}_{yz} + D_{1} \, \mathcal{L}^{y}_{zx}$ |
# | $C_{2}$ | $D_{0} \, \mathcal{L}^{x}_{zx} + D_{1} \, \mathcal{L}^{y}_{yz} + D_{2} \, \mathcal{L}^{x}_{yz} + D_{3} \, \mathcal{L}^{y}_{zx}$ |
# | $C_{1h}$ | $D_{0} \, \mathcal{L}^{x}_{xy} + D_{1} \, \mathcal{L}^{y}_{xy}$ |
# | $C_{1v}$ | $D_{0} \, \mathcal{L}^{x}_{zx} + D_{1} \, \mathcal{L}^{y}_{xy} + D_{2} \, \mathcal{L}^{y}_{yz}$ |
# | $C_{1}$ | $D_{0} \, \mathcal{L}^{x}_{xy} + D_{1} \, \mathcal{L}^{x}_{yz} + D_{2} \, \mathcal{L}^{x}_{zx} + D_{3} \, \mathcal{L}^{y}_{xy} + D_{4} \, \mathcal{L}^{y}_{yz} + D_{5} \, \mathcal{L}^{y}_{zx}$ |

# ### Using Predefined DMI Fields
#
# NeuralMag provides predefined fields for common DMI types, such as bulk DMI.

mesh = nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)

state.m = nm.VectorFunction(state).fill((0, 0, 1))

state.material.Ms = 0.86e6
state.material.alpha = 1
state.material.A = 1.3e-11
nm.ExchangeField().register(state, "exchange")


state.material.Db = 15e-3
nm.BulkDMIField().register(state, "bulk")

nm.TotalField("exchange", "bulk").register(state)


# ### Equivalent Using Multiple Lifshitz Invariants

mesh = nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)

state.m = nm.VectorFunction(state).fill((0, 0, 1))

state.material.Ms = 0.86e6
state.material.alpha = 1
state.material.A = 1.3e-11
nm.ExchangeField().register(state, "exchange")


state.material.Dxyz = 15e-3
state.material.Dzxy = state.material.Dxyz
state.material.Dyzx = state.material.Dxyz
nm.LIField("xyz").register(state, "li_xyz")
nm.LIField("zxy").register(state, "li_zxy")
nm.LIField("yzx").register(state, "li_yzx")

nm.TotalField("exchange", "li_xyz", "li_zxy", "li_yzx").register(state)


# ### Flexibility with Independent Constants
#
# By setting different values for the DMI constants, we can reduce the symmetry, equivalent to changing from high-symmetry classes (like T or O) to lower-symmetry ones (like D2).

mesh = nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)

state.m = nm.VectorFunction(state).fill((0, 0, 1))

state.material.Ms = 0.86e6
state.material.alpha = 1
state.material.A = 1.3e-11
nm.ExchangeField().register(state, "exchange")


state.material.Dxyz = 15e-3
state.material.Dzxy = 20e-3
state.material.Dyzx = 10e-3
nm.LIField("xyz").register(state, "li_xyz")
nm.LIField("zxy").register(state, "li_zxy")
nm.LIField("yzx").register(state, "li_yzx")

nm.TotalField("exchange", "li_xyz", "li_zxy", "li_yzx").register(state)


# ## Summary
#
# * Lifshitz invariants are the building blocks of DMI.
# * By adding the right set of invariants, we represent the DMI for a given crystal symmetry.

#
