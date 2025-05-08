# SPDX-License-Identifier: MIT

name = "jax"

import neuralmag.backends.jax.demag_field
from neuralmag.backends.jax.code_generation import *
from neuralmag.backends.jax.llg_solver_jax import *
from neuralmag.backends.jax.tensor_operations import *

LLGSolver = LLGSolverJAX
