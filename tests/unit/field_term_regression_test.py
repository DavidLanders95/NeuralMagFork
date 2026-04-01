import numpy as np
import pytest
from scipy import constants

from neuralmag import *

be = config.backend


# --- State constructors ---


def _make_state(disc):
    """Create a state with non-uniform magnetization for the given discretization."""
    if disc == "fem1d":
        mesh = Mesh((2,), (1e-9, 1e-9, 1e-9))
        state = State(mesh)
        m = np.zeros((3, 3))
        m[0, 0] = -1
        m[1, 1] = 1
        m[2, 0] = 1
        state.m = VectorFunction(state, tensor=state.tensor(m))
    elif disc == "fem2d":
        mesh = Mesh((2, 2), (1e-9, 1e-9, 1e-9))
        state = State(mesh)
        m = np.zeros((3, 3, 3))
        m[0, :, 0] = -1
        m[1, :, 1] = 1
        m[2, :, 0] = 1
        state.m = VectorFunction(state, tensor=state.tensor(m))
    elif disc == "fem3d":
        mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
        state = State(mesh)
        m = np.zeros((3, 3, 3, 3))
        m[0, :, :, 0] = -1
        m[1, :, :, 1] = 1
        m[2, :, :, 0] = 1
        m[1, 0, 0, :] = 0
        m[1, 0, 0, 0] = 1
        state.m = VectorFunction(state, tensor=state.tensor(m))
    elif disc == "fic":
        mesh = Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9))
        state = State(mesh)
        m = np.zeros((2, 2, 2, 3))
        m[0, :, :, 0] = -1
        m[1, :, :, 1] = 1
        state.m = VectorCellFunction(state, tensor=state.tensor(m))
    return state


def _set_material(state, params):
    """Set material parameters on state."""
    for k, v in params.items():
        if isinstance(v, list):
            setattr(state.material, k, VectorCellFunction(state).fill(v))
        else:
            setattr(state.material, k, CellFunction(state).fill(v))


# --- Material parameters per field term ---

MATERIAL = {
    ExchangeField: {"A": 1.2e-11, "Ms": 8e5},
    BulkDMIField: {"Db": 1e-3, "Ms": 8e5},
    InterfaceDMIField: {"Di": 1e-3, "Di_axis": [0, 0, 1], "Ms": 8e5},
    UniaxialAnisotropyField: {"Ku": 1e6, "Ku_axis": [0, 1, 0], "Ms": 8e5},
    ExternalField: {"Ms": 8e5},
}

# --- Reference values: {(FieldTermClass, disc): {h_sum, e_sum, E}} ---

REFS = {
    (ExchangeField, "fem1d"): {"h_sum": 47746484.0, "e_sum": 72000000.0, "E": 4.8e-20},
    (ExchangeField, "fem2d"): {
        "h_sum": 143239472.0,
        "e_sum": 216000000.0,
        "E": 9.6e-20,
    },
    (ExchangeField, "fem3d"): {
        "h_sum": 429718432.0,
        "e_sum": 657500032.0,
        "E": 1.96e-19,
    },
    (ExchangeField, "fic"): {"h_sum": 0.0, "e_sum": 48000000.0, "E": 4.8e-20},
    (BulkDMIField, "fem1d"): {"h_sum": 0.0, "e_sum": 0.0, "E": 0.0},
    (BulkDMIField, "fem2d"): {"h_sum": 0.0, "e_sum": 0.0, "E": 0.0},
    (BulkDMIField, "fem3d"): {
        "h_sum": 1105240.75,
        "e_sum": -1041666.75,
        "E": -3.333e-22,
    },
    (BulkDMIField, "fic"): {"h_sum": -3978874.0, "e_sum": 0.0, "E": 0.0},
    (InterfaceDMIField, "fem1d"): {"h_sum": 1989436.75, "e_sum": 0.0, "E": 0.0},
    (InterfaceDMIField, "fem2d"): {"h_sum": 5968310.5, "e_sum": 0.0, "E": 0.0},
    (InterfaceDMIField, "fem3d"): {"h_sum": 19010176.0, "e_sum": 0.0, "E": 0.0},
    (InterfaceDMIField, "fic"): {"h_sum": 3978873.5, "e_sum": 0.0, "E": 0.0},
    (UniaxialAnisotropyField, "fem1d"): {
        "h_sum": 2652582.5,
        "e_sum": -833333.4375,
        "E": -6.667e-22,
    },
    (UniaxialAnisotropyField, "fem2d"): {
        "h_sum": 7957746.5,
        "e_sum": -2500000.25,
        "E": -1.333e-21,
    },
    (UniaxialAnisotropyField, "fem3d"): {
        "h_sum": 22031168.0,
        "e_sum": -6626156.5,
        "E": -2.407e-21,
    },
    (UniaxialAnisotropyField, "fic"): {
        "h_sum": 7957747.0,
        "e_sum": -2666667.0,
        "E": -2.667e-21,
    },
    (ExternalField, "fem1d"): {"h_sum": 18.0, "e_sum": -2.6808252, "E": -2.011e-27},
    (ExternalField, "fem2d"): {"h_sum": 54.0, "e_sum": -8.0424767, "E": -4.021e-27},
    (ExternalField, "fem3d"): {"h_sum": 162.0, "e_sum": -23.196583, "E": -7.791e-27},
    (ExternalField, "fic"): {"h_sum": 48.0, "e_sum": -4.0212388, "E": -4.021e-27},
}


def _id(key):
    cls, disc = key
    return f"{cls.__name__}-{disc}"


@pytest.mark.parametrize("key", REFS.keys(), ids=[_id(k) for k in REFS])
def test_field_term(key):
    field_cls, disc = key
    ref = REFS[key]

    state = _make_state(disc)
    _set_material(state, MATERIAL[field_cls])

    # ExternalField needs explicit h vector
    if field_cls is ExternalField:
        if disc == "fic":
            h = VectorCellFunction(state).fill([1.0, 2.0, 3.0])
        else:
            h = VectorFunction(state).fill([1.0, 2.0, 3.0])
        field_cls(h).register(state)
    else:
        field_cls().register(state)

    name = field_cls.default_name
    h_sum = float(be.to_numpy(getattr(state, f"h_{name}").tensor.sum()))
    e_sum = float(be.to_numpy(getattr(state, f"e_{name}").tensor.sum()))
    E = float(be.to_numpy(getattr(state, f"E_{name}")))

    assert h_sum == pytest.approx(ref["h_sum"], rel=1e-3, abs=20.0)
    assert e_sum == pytest.approx(ref["e_sum"], rel=1e-3, abs=1.0)
    assert E == pytest.approx(ref["E"], rel=1e-2, abs=1e-28)
