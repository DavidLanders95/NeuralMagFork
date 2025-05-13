#!/usr/bin/env python
# coding: utf-8

# # Standard problem 4
#
# ## Problem specification
#
# The sample is a thin film cuboid with dimensions:
#
# - length $l_{x} = 500 \,\text{nm}$,
# - width $l_{y} = 125 \,\text{nm}$, and
# - thickness $l_{z} = 3 \,\text{nm}$.
#
# The material parameters (similar to permalloy) are:
#
# - exchange energy constant $A = 1.3 \times 10^{-11} \,\text{J/m}$,
# - magnetisation saturation $M_\text{s} = 8 \times 10^{5} \,\text{A/m}$.
#
# Magnetisation dynamics are governed by the Landau-Lifshitz-Gilbert equation
#
# $$\frac{d\mathbf{m}}{dt} = \underbrace{-\gamma_{0}(\mathbf{m} \times \mathbf{H}_\text{eff})}_\text{precession} + \underbrace{\alpha\left(\mathbf{m} \times \frac{d\mathbf{m}}{dt}\right)}_\text{damping}$$
#
# where $\gamma_{0} = 2.211 \times 10^{5} \,\text{m}\,\text{A}^{-1}\,\text{s}^{-1}$ and Gilbert damping $\alpha=0.02$.
#
# In the standard problem 4, the system is first relaxed at zero external magnetic field and then, starting from the obtained equlibrium configuration, the magnetisation dynamics are simulated for two external magnetic fields $\mathbf{B}_{1} = (-24.6, 4.3, 0.0) \,\text{mT}$ and $\mathbf{B}_{2} = (-35.5, -6.3, 0.0) \,\text{mT}$.
#
# More detailed specification of Standard problem 4 can be found in Ref. 1.
#
# ## Simulation
#
# ### Import modules
# In the first step, we import the required modules and configure pyvista for static rendering.

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy import constants

import neuralmag as nm

pv.set_jupyter_backend("static")


# ### Setup mesh and state
#
# In the first stage, we need to setup the mesh and the simulation state. We chose a cell size of $5 \times 5 \times 3 \,\text{nm}^3$ resulting in $100 \times 25 \times 1$ cells simulate for the geometry defined for this standard problem.

mesh = nm.Mesh([100, 25, 1], [5e-9, 5e-9, 3e-9], [0.0, 0.0, 0.0])
state = nm.State(mesh)


# ### Setup material parameters and define initial magnetization
# In the next step, we set the material parameters $M_s$, $A$ and $\alpha$ according to the requirements of the standard problem.

state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.alpha = 0.02


# We have to provide an initial magnetisation configuration that is going to be relaxed subsequently. We choose the uniform configuration in xy-direction that is know to relax into the requires s-state.

state.m = nm.VectorFunction(state).fill((0.5**0.5, 0.5**0.5, 0))


# ### Register effectiv-field contributions
# Now we initialize the effecitive field contributions that are required for the relaxation of the initial magnetization. Namely, we set up a total field consisting of the exchange field and the demagnetization field.

nm.ExchangeField().register(state, "exchange")
nm.DemagField().register(state, "demag")
nm.TotalField("exchange", "demag").register(state)


# ### Minimize energy to find the initial stable state
# We initialze the LLG solver and use the ```relax``` method in order relax the system into the stable s-state configuration.

llg = nm.LLGSolver(state)
llg.relax()
state.write_vti(["m"], "standard-problem-4/s-state.vti")

# Plot the flower state
mesh = pv.read("standard-problem-4/s-state.vti")
glyphs = mesh.glyph(orient="m", scale="m", factor=1e-8)
p = pv.Plotter()
p.add_mesh(glyphs, color="white", lighting=True, smooth_shading=True)
p.show()


# ### Apply external field
# In the next step we initialize the an external field to switch the magnetization as defined in the standard problem. In order to include this additional field, we update the total field and reset the LLG solver.

# Setup Zeeman field
h_ext = nm.VectorFunction(state).fill(
    [-24.6e-3 / constants.mu_0, 4.3e-3 / constants.mu_0, 0.0]  # B1
    # [-35.5e-3 / constants.mu_0, -6.3e-3 / constants.mu_0, 0.0]  # B2
)
nm.ExternalField(h_ext).register(state, "external")

# Update total field
nm.TotalField("exchange", "demag", "external").register(state)

# Reset LLGSolver
llg.reset()


# ### Simulate switching
# Finally, we can run the switching simulation using the LLGSolver. We run the magnetisation evolution for $t=1 \,\text{ns}$. We set up a logger to save the averaged magnetization as well as a time series of the full magnetization configuration and log the simulation state every $10\,\text{ps}$.

logger = nm.Logger("standard-problem-4", ["t", "m"], ["m"])
while state.t < 1e-9:
    llg.step(1.0e-11)
    logger.log(state)


# ### Postprocessing

# Finally, we want to plot the average magnetization configuration as a function of time `t`:

data = np.loadtxt("standard-problem-4/log.dat")
plt.plot(data[:, 0], data[:, 1], label="m_x")
plt.plot(data[:, 0], data[:, 2], label="m_y")
plt.plot(data[:, 0], data[:, 3], label="m_z")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("m_i")
plt.show()


# ## References
#
# [1] µMAG Site Directory: http://www.ctcms.nist.gov/~rdm/mumag.org.html
#
# This tutorial was adapted from [Ubermag](https://ubermag.github.io/).
