from neuralmag import *
from scipy import constants

config.fem['n_gauss'] = 1

mesh = Mesh((2, 2, 2), (5e-9, 5e-9, 5e-9))
state = State(mesh)

# setup material and m0
state.material.Ms = 8e5
state.material.A = 1.3e-11
state.material.Ku = 1e5
state.material.Ku_axis = [0, 0, 1]
state.material.alpha = 1.

state.m = VectorFunction(state).from_constant((0, 0, 1))

# setup external field depending t
Hc = 2 * 1e5 / (constants.mu_0 * 8e5)
h_ext = lambda t: t * state.tensor([0, 2 / 50e-9 * Hc, 0]).reshape((1,1,1,3)).expand((3,3,3,3))

# register effective field
ExchangeField().register(state, 'exchange')
UniaxialAnisotropyField().register(state, 'aniso')
ExternalField(h_ext).register(state, 'external')
TotalField('exchange', 'aniso', 'external').register(state)

# relax
llg = LLGSolver(state)

# foward
logger = Logger('sw', ['t', 'h_external', 'm', 'E'], ['m'])
while state.t < 50e-9:
    logger.log(state)
    llg.step(1e-11)
