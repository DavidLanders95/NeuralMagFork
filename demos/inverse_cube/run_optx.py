###############################################################################
###
### Simple dynamic inverse example to find the optimum field angle in order
### achieve a given magnetization tilting of a single-domain particle after
### 0.5 ns.
###
### Uses the JAX backend with the BFGS minimizer of the optimistix lib
###
###############################################################################

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from scipy import constants

import neuralmag as nm

nm.config.backend = "jax"
nm.config.fem["n_gauss"] = 1

mesh = nm.Mesh((2, 2, 2), (5e-9, 5e-9, 5e-9))
state = nm.State(mesh)

# setup material and m0
state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.Ku = 1e5
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 0.1

state.m = nm.VectorFunction(state).fill((0, 0, 1))

# setup external field depending on phi and theta
Hc = 2 * 1e5 / (constants.mu_0 * 8e5)
state.phi = lambda angles: angles[0]
state.theta = lambda angles: angles[1]
state.angles = [jnp.pi / 2, jnp.pi / 2]
h_ext = lambda angles: jnp.stack(
    [
        Hc / 2 * jnp.sin(angles[0]) * jnp.cos(angles[1]),
        Hc / 2 * jnp.sin(angles[0]) * jnp.sin(angles[1]),
        Hc / 2 * jnp.cos(angles[0]),
    ]
)

# register effective field
nm.ExchangeField().register(state, "exchange")
nm.UniaxialAnisotropyField().register(state, "aniso")
nm.ExternalField(h_ext).register(state, "external")
nm.TotalField("exchange", "aniso", "external").register(state)

# set up solver, loss function, etc
llg = nm.LLGSolver(state, parameters=["angles"])

m_target = nm.VectorFunction(state).fill((0.5**0.5, 0, 0.5**0.5)).tensor


@eqx.filter_jit
def loss(angles, args):
    m_pred = llg.solve(state.tensor([0.0, 0.05e-9]), angles).ys[-1]
    return jnp.mean((m_target - m_pred) ** 2)


solver = optx.BFGS(1e-3, 1e-3, optx.max_norm)
result = optx.minimise(loss, solver, state.angles)

print(result.value)
