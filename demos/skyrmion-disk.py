#!/usr/bin/env python
# coding: utf-8

# # Skyrmion in a disk

# In this example, we stabilize a single Skyrmion in a circular shaped thin film.
#
# ## Simulation
#
# ### Import libraries

import pyvista as pv
from scipy import constants

import neuralmag as nm
from _static_method_compare import SOLVER_LABELS, compare_static_methods, print_static_method_summary

pv.set_jupyter_backend("static")

METHODS = ("llg", "bb")
SELECTED_METHOD = "bb"


# ### Create mesh and state
# We create a 2D nodal mesh (mesh with just 1 layer of nodes in the z-direction) with $50 \times 50$ cells with cell size $2 \times 2 \times 0.6\,\text{nm}^3$.

def build_state():
	mesh = nm.Mesh((50, 50), (2e-9, 2e-9, 0.6e-9), (-50e-9, -50e-9, 0))
	state = nm.State(mesh)

	state.material.Ms = 1.0 / constants.mu_0
	state.material.A = 1.6e-11
	state.material.Di = 4e-3
	state.material.Di_axis = [0, 0, 1]
	state.material.Ku = 510e3
	state.material.Ku_axis = [0, 0, 1]
	state.material.alpha = 0.1

	x, y = state.coordinates()
	state.add_domain(1, x**2.0 + y**2.0 < 50e-9**2.0)

	state.m = nm.VectorFunction(state).fill((0, 0, 1))

	nm.ExchangeField().register(state, "exchange")
	nm.DemagField().register(state, "demag")
	nm.InterfaceDMIField().register(state, "dmi")
	nm.UniaxialAnisotropyField().register(state, "aniso")
	nm.TotalField("exchange", "demag", "dmi", "aniso").register(state)

	return state


comparison = compare_static_methods(
	build_state,
	methods=METHODS,
	llg_builder=lambda state: nm.LLGSolver(state, scale_t=1e-12),
	llg_runner=lambda solver: solver.relax(1e9),
)
print_static_method_summary("Skyrmion disk", comparison)

for method, values in comparison.items():
	values["state"].write_vti(["m", "rho", "e"], f"skyrmion-disk/m-{method}.vti")


# ## Visualization
# We use pyvista to visualize the resulting skyrmion, using a threshold filte on ```state.rho``` in order to show only the magnetic region.

plotter = pv.Plotter(shape=(1, len(METHODS)))
for column, method in enumerate(METHODS):
	grid = pv.read(f"skyrmion-disk/m-{method}.vti")
	thresholded = grid.threshold(value=0.5, scalars="rho", preference="cell")
	glyphs = thresholded.glyph(orient="m", scale="m", factor=1e-8)
	plotter.subplot(0, column)
	plotter.add_text(SOLVER_LABELS[method], font_size=12)
	plotter.add_mesh(glyphs, scalars="e", lighting=True, smooth_shading=True)

plotter.link_views()
plotter.show()
