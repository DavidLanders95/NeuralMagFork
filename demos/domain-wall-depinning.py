#!/usr/bin/env python
# coding: utf-8

# # Domain-wall pinning at material interface
# This example implements the benchmark problem proposed in [1] where a domain wall is pinned at the material interface of a magnetic wire.
#
# ## Simulation
# ### Import libraries
# Import libraries and set global config options.

import numpy as np
from scipy import constants

import neuralmag as nm

nm.config.fem["n_gauss"] = 1
nm.config.dtype = "float64"


# ### Set up mesh and state

mesh = nm.Mesh((80,), (1e-9, 1e-9, 1e-9))
state = nm.State(mesh)


# ### Set up material parameters
# Set up material parameters in the wire with optional material jumps in the satureation magnetization $M_s$, the exchange constant $A$ and or the anisotropy constant $K$. The material parameters are prepared in a NumPy array and then copied to the respective material functions. The ```mode``` string indicates the parameters with jump ("m"/"a"/"k" leads to a jump in $M_s$, $A$ or $K$ respectively and "ak" lead to a jump in both $K$ and $A$.

MODE = "a"

Ms = np.ones(80) * (1.0 / constants.mu_0)
A = np.ones(80) * (1e-11)
Ku = np.ones(80) * (1e6)

if "m" in MODE:
    Ms[:40] = 0.25 / constants.mu_0
if "a" in MODE:
    A[:40] = 0.25e-11
if "k" in MODE:
    Ku[:40] = 1e5

state.material.Ms = nm.CellFunction(state, tensor=state.tensor(Ms))
state.material.A = nm.CellFunction(state, tensor=state.tensor(A))
state.material.Ku = nm.CellFunction(state, tensor=state.tensor(Ku))
state.material.Ku_axis = [0, 1, 0]
state.material.alpha = 1.0


# ### Initialize the magnetization
# Now, we initialize the magnetization with a parametrized domain wall next to the material boundary.

m = np.zeros((81, 3))
m[:35, 1] = np.cos(0.3)
m[:35, 0] = np.sin(0.3)
m[35:, 1] = -1.0
state.m = nm.VectorFunction(state, tensor=state.tensor(m))


# ### Register effective field
# We register an effective field comprised by uniaxial anisotropy, exchange and an external field in y-direction that linearly increases in time.

nm.UniaxialAnisotropyField().register(state, "aniso")
nm.ExchangeField().register(state, "exchange")
hrate = state.tensor([0, 1.6 / constants.mu_0 / 100e-9, 0])
nm.ExternalField(lambda t: t * hrate).register(state, "external")
nm.TotalField("aniso", "exchange", "external").register(state)


# ### Perform time integration
# Finally we perform time integration observing the y-component of the averaged magnetization as an indicator for depinning. Once the value exceeds 0.55, we assume that the domain-wall has depinned and we print the strength of the depinning field.

llg = nm.LLGSolver(state, max_steps=None)
state.t = 20e-9
while state.t < 100e-9:
    llg.step(1e-9)
    if state.m.avg()[1] > 0.55:
        print(f"Depinning Field H = {state.h_external.avg()[1] * constants.mu_0} A/m")
        break


# ## References
# [1] Heistracher, P., Abert, C., Bruckner, F., Schrefl, T., & Suess, D. (2022). Proposal for a micromagnetic standard problem: domain wall pinning at phase boundaries. Journal of Magnetism and Magnetic Materials, 548, 168875.
