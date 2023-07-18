import numpy as np

from nmagnum import *

mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
state = State(mesh)

state.material.Ms = CellFunction(state).from_constant(8e5)
state.material.A = CellFunction(state).from_constant(1.3e-12)
state.material.alpha = 1.0

state.m = VectorFunction(state).from_constant(
    [np.cos(np.radians(10)), np.sin(np.radians(10)), 0]
)

exchange = ExchangeTorchField()
demag = DemagField()

llg = LLGSolver([demag, exchange])

while state.t < 1e-9:
    llg.step(state, 1e-12)

state.m.write("m.vti")
# state.material.Ms.write("ms.vti")
