import argparse
import time

import torch
from diffrax import Dopri5
from scipy import constants

import neuralmag
from neuralmag import *

solver = Dopri5()
evals_per_step = solver.tableau.num_stages - solver.tableau.fsal * 1

parser = argparse.ArgumentParser()
parser.add_argument("e", type=int)
args = parser.parse_args()

N = 2**args.e
print("N:", N)

config.backend = "jax"

### Init
# setup mesh and state
mesh = Mesh((N, N), (4e-9, 4e-9, 4e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = 800e3
state.material.A = 13e-12
state.material.alpha = 0.01

# initialize nodal vector functions for magneization and external field
state.m = VectorFunction(state).fill((1, 0, 0))
h_ext = VectorFunction(state).fill((0, 0.01 / constants.mu_0, 0), expand=True)

# register effective field contributions
ExchangeField().register(state, "exchange")
DemagField().register(state, "demag")
ExternalField(h_ext).register(state, "external")
TotalField("exchange", "demag", "external").register(state)

llg = LLGSolverJAX(state)
llg.step(1e-13)


### Warmup
llg.step(1e-12)

### Measure
start = time.perf_counter()
sol = llg.step(1e-10)
t_total = time.perf_counter() - start
n_eval = int(sol.stats["num_steps"]) * evals_per_step

# n_eval = Timer._timers['###measure###LLGSolver.step###DemagField.h']['calls']

print(
    "NN=",
    N * N,
    "n_eval=",
    n_eval,
    "t_total=",
    t_total,
    "throuhput=",
    N * N * n_eval / t_total,
)
