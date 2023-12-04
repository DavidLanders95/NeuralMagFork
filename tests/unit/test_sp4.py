from neuralmag import *
import pytest
import pandas as pd

@pytest.fixture(scope="function")
def s_state():
    mesh = Mesh([100, 25, 3], [5e-09, 5e-09, 1e-09], [0.0, 0.0, 0.0])
    state = State(mesh)

    state.material.Ms = 800000.0
    state.material.A = 1.3e-11
    state.material.alpha = 1
    h_ext = VectorFunction(state).from_constant((0, 0, 0))

    # initial s-state
    state.m = state.read_vti("tests/unit/s_state_init.vti", 'm')

    ExchangeField().register(state, "exchange")
    DemagField().register(state, "demag")
    ExternalField(h_ext).register(state, "external")
    TotalField("exchange", "demag", "external").register(state)
    # Make sure s-state is relaxed
    llg = LLGSolver(state)
    llg.step(1e-10)
    
    return state

def test_sp4_field_switch_1(s_state):
    # set external field and perform switch
    s_state.material.alpha = 0.02
    s_state.h_external = VectorFunction(s_state).from_constant((-19576.05800030313, 3421.8312764757497, 0.0))

    llg = LLGSolver(s_state)

    logger = Logger('data', ['t', 'm'], ['m'])
    for _ in range(30):
        llg.step(10.0e-12)
        logger.log(s_state)
        
    data = pd.read_csv(
        "data/log.dat",
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t", "m_x", "m_y", "m_z"]
    )
    
    max_time = data["t"][data["m_y"].idxmax()]
    max_y = data["m_y"].max()
    
    assert (max_time >= 2.1e-10) & (max_time <= 2.4e-10)
    assert (max_y >= 0.67) & (max_y <= 0.77)
    
    min_time = data["t"][data["m_y"].idxmin()]
    min_y = data["m_y"].min()

    assert (min_time >= 3.2e-10) & (min_time <= 3.5e-10)
    assert (min_y >= -0.5) & (min_y <= -0.4)

    
        
    