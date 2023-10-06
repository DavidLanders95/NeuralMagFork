from nmagnum import *
import numpy as np

# setup state
mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-11)
state.material.alpha = 1.

state.m = VectorFunction(state).from_constant((np.sqrt(0.5), np.sqrt(0.5), 0))
h_ext = VectorFunction(state).from_constant((0, 0, 0))

# register effective field
ExchangeField().register(state, 'exchange')
DemagField().register(state, 'demag')
ExternalField(h_ext).register(state, 'external')
TotalField('exchange', 'demag', 'external').register(state)

# relax to s-state
llg = LLGSolver(state)
llg.step(1e-9)

# set external field and perform switch
h_ext.tensor[..., :] = state.tensor([-19576., 3421., 0.])
state.material.alpha = 0.02

logger = Logger('data', ['t', 'm'], ['m'])
for i in range(100):
    logger.log(state)
    llg.step(1e-11)
