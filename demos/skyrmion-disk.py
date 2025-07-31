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

pv.set_jupyter_backend("static")


# ### Create mesh and state
# We create a 2D nodal mesh (mesh with just 1 layer of nodes in the z-direction) with $50 \times 50$ cells with cell size $2 \times 2 \times 0.6\,\text{nm}^3$.

mesh = nm.Mesh((50, 50), (2e-9, 2e-9, 0.6e-9), (-50e-9, -50e-9, 0))
state = nm.State(mesh)


# ### Set material parameters

state.material.Ms = 1.0 / constants.mu_0
state.material.A = 1.6e-11
state.material.Di = 4e-3
state.material.Di_axis = [0, 0, 1]
state.material.Ku = 510e3
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 0.1


# ### Define circular geometry
# Initialize geometry by defining a circular domain with ID 1.

x, y = state.coordinates()
state.add_domain(1, x**2.0 + y**2.0 < 50e-9**2.0)


# ### Set initial magnetization

state.m = nm.VectorFunction(state).fill((0, 0, 1))


# ### Register effective field
# The effective field comprises exchange, demag, interface DMI and uniaxial anisotropy contributions

nm.ExchangeField().register(state, "exchange")
nm.DemagField().register(state, "demag")
nm.InterfaceDMIField().register(state, "dmi")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.TotalField("exchange", "demag", "dmi", "aniso").register(state)


# ### Relax to skyrmion configuration

llg = nm.LLGSolver(state, scale_t=1e-12)
llg.relax(1e9)
state.write_vti(["m", "rho"], "skyrmion-disk/m.vti")


# ## Visualization
# We use pyvista to visualize the resulting skyrmion, using a threshold filte on ```state.rho``` in order to show only the magnetic region.

grid = pv.read("skyrmion-disk/m.vti")

thresholded = grid.threshold(value=0.5, scalars="rho", preference="cell")
glyphs = thresholded.glyph(orient="m", scale="m", factor=1e-8)

# Plot
plotter = pv.Plotter()
plotter.add_mesh(glyphs, color="white", lighting=True, smooth_shading=True)
plotter.show()
