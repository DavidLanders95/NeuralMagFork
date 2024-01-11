import copy
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from neuralmag import *


# How the s_state_init.vti file was produced
@pytest.fixture(scope="function")
def s_state():
    mesh = Mesh([100, 25, 1], [5e-09, 5e-09, 3e-09], [0.0, 0.0, 0.0])
    state = State(mesh)

    state.material.Ms = 800000.0
    state.material.A = 1.3e-11
    state.material.alpha = 1
    h_ext = VectorFunction(state).from_constant((0, 0, 0))
    state.m = VectorFunction(state).from_constant((1, 0.25, 0.1))

    ExchangeField().register(state, "exchange")
    DemagField().register(state, "demag")
    ExternalField(h_ext).register(state, "external")
    TotalField("exchange", "demag", "external").register(state)
    # Make sure s-state is relaxed
    llg = LLGSolver(state)
    llg.step(10e-10)

    state.t = 0.0

    return state


def test_sp4_field_switch_1(s_state):
    s_state.material.alpha = 0.02
    s_state.h_external = VectorFunction(s_state).from_constant(
        (-19576.05800030313, 3421.8312764757497, 0.0)
    )

    llg = LLGSolver(s_state)

    logger = Logger("data", ["t", "m"], ["m"])
    for _ in range(30):
        llg.step(1.0e-11)
        logger.log(s_state)

    data_oommf = pd.read_csv(
        "tests/unit/data/oommf_sp4_fieldswitch_1.csv",
    )

    data = pd.read_csv(
        "data/log.dat",
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t", "m_x", "m_y", "m_z"],
    )

    difference = data["m_y"] - np.interp(data["t"], data_oommf["t"], data_oommf["m_y"])

    try:
        assert all(abs(difference) < 0.04)
    except AssertionError:
        plt.plot(data["t"], data["m_y"], label="neural_mag")
        plt.plot(data_oommf["t"], data_oommf["m_y"], label="oommf")
        plt.legend()
        plt.xlim([0, 3e-10])
        plt.xlabel("t (s)")
        plt.ylabel("my (normalised)")
        plt.savefig("tests/test_artifacts/sp4_failure_plot_field_switch_1.png")
        raise


def test_sp4_field_switch_2(s_state):
    s_state.material.alpha = 0.02
    s_state.h_external = VectorFunction(s_state).from_constant(
        (-28250.00239881142, -5013.380707394703, 0.0)
    )

    llg = LLGSolver(s_state)

    logger = Logger("data", ["t", "m"], ["m"])
    for _ in range(30):
        llg.step(1.0e-11)
        logger.log(s_state)

    data_oommf = pd.read_csv(
        "tests/unit/data/oommf_sp4_fieldswitch_2.csv",
    )

    data = pd.read_csv(
        "data/log.dat",
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t", "m_x", "m_y", "m_z"],
    )

    difference = data["m_y"] - np.interp(data["t"], data_oommf["t"], data_oommf["m_y"])

    try:
        assert all(abs(difference) < 0.04)
    except AssertionError:
        plt.plot(data["t"], data["m_y"], label="neural_mag")
        plt.plot(data_oommf["t"], data_oommf["m_y"], label="oommf")
        plt.legend()
        plt.xlim([0, 3e-10])
        plt.xlabel("t (s)")
        plt.ylabel("my (normalised)")
        plt.savefig("tests/test_artifacts/sp4_failure_plot_field_switch_2.png")
        raise
