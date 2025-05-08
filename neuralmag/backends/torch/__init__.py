# SPDX-License-Identifier: MIT

name = "torch"

import neuralmag.backends.torch.demag_field
from neuralmag.backends.torch.code_generation import *
from neuralmag.backends.torch.llg_solver_torch import *
from neuralmag.backends.torch.tensor_operations import *

LLGSolver = LLGSolverTorch
