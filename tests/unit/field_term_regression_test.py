# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from scipy import constants

from neuralmag import *

be = config.backend


# --- Magnetization pattern ---


def _make_m(shape):
    """Helix-like magnetization that breaks all spatial symmetries.

    Every node/cell gets a unique direction so that all field terms
    (exchange, DMI, anisotropy) produce nonzero h, e, and E.
    """
    m = np.zeros(shape + (3,), dtype=np.float32)
    for idx in np.ndindex(*shape):
        t = sum(i * (2**d) for d, i in enumerate(idx)) * 0.7
        v = np.array([np.cos(t), np.sin(t), 0.3 * np.sin(2 * t)])
        m[idx] = v / np.linalg.norm(v)
    return m


# --- State constructors ---


def _make_state(disc):
    if disc == "fem1d":
        state = State(Mesh((2,), (1e-9, 1e-9, 1e-9)))
        state.m = VectorFunction(state, tensor=state.tensor(_make_m((3,))))
    elif disc == "fem2d":
        state = State(Mesh((2, 2), (1e-9, 1e-9, 1e-9)))
        state.m = VectorFunction(state, tensor=state.tensor(_make_m((3, 3))))
    elif disc == "fem3d":
        state = State(Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9)))
        state.m = VectorFunction(state, tensor=state.tensor(_make_m((3, 3, 3))))
    elif disc == "fic1d":
        state = State(Mesh((2,), (1e-9, 1e-9, 1e-9)))
        state.m = VectorCellFunction(state, tensor=state.tensor(_make_m((2,))))
    elif disc == "fic2d":
        state = State(Mesh((2, 2), (1e-9, 1e-9, 1e-9)))
        state.m = VectorCellFunction(state, tensor=state.tensor(_make_m((2, 2))))
    elif disc == "fic":
        state = State(Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9)))
        state.m = VectorCellFunction(state, tensor=state.tensor(_make_m((2, 2, 2))))
    elif disc == "fem3d_pbc":
        state = State(Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9), pbc=True))
        state.m = VectorFunction(state, tensor=state.tensor(_make_m((2, 2, 2))))
    elif disc == "fic_pbc":
        state = State(Mesh((2, 2, 2), (1e-9, 1e-9, 1e-9), pbc=True))
        state.m = VectorCellFunction(state, tensor=state.tensor(_make_m((2, 2, 2))))
    return state


def _set_material(state, params):
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
    # fmt: off
    #                                                        h_sum               e_sum            E
    (ExchangeField,          "fem1d"):     {"h_sum":   24347980.0,    "e_sum":   18302580.0,    "E":  1.22e-20},
    (ExchangeField,          "fem2d"):     {"h_sum":    4388404.0,    "e_sum":  208698960.0,    "E":  9.30e-20},
    (ExchangeField,          "fem3d"):     {"h_sum": -321162432.0,    "e_sum": 1052602112.0,    "E":  3.117e-19},
    (ExchangeField,          "fic1d"):     {"h_sum":       -0.75,    "e_sum":    3198466.0,    "E":  3.198e-21},
    (ExchangeField,          "fic2d"):     {"h_sum":         1.0,    "e_sum":   23044290.0,    "E":  2.304e-20},
    (ExchangeField,          "fic"):       {"h_sum":         1.5,    "e_sum":   77500352.0,    "E":  7.75e-20},
    (ExchangeField,          "fem3d_pbc"): {"h_sum":        32.0,    "e_sum":  310001408.0,    "E":  3.10e-19},
    (ExchangeField,          "fic_pbc"):   {"h_sum":         0.0,    "e_sum":          0.0,    "E":  0.0},       # cell→node projection gives uniform nodes → physical zero

    (BulkDMIField,           "fem1d"):     {"h_sum":   -875867.4,    "e_sum":    324308.75,    "E":  2.162e-22},
    (BulkDMIField,           "fem2d"):     {"h_sum":  -3083332.5,    "e_sum":   -426271.5,     "E": -1.885e-22},
    (BulkDMIField,           "fem3d"):     {"h_sum":  -2142686.25,   "e_sum":  -5660799.0,     "E": -1.691e-21},
    (BulkDMIField,           "fic1d"):     {"h_sum":   -332515.125,  "e_sum":          0.0,    "E":  0.0},
    (BulkDMIField,           "fic2d"):     {"h_sum":  -2146756.5,    "e_sum":    108385.25,    "E":  1.084e-22},
    (BulkDMIField,           "fic"):       {"h_sum":   -625226.0,    "e_sum":   -895176.625,   "E": -8.952e-22},
    (BulkDMIField,           "fem3d_pbc"): {"h_sum":         0.0,    "e_sum":          0.0,    "E":  0.0},       # PBC + helix: DMI vanishes by symmetry
    (BulkDMIField,           "fic_pbc"):   {"h_sum":         0.0,    "e_sum":          0.0,    "E":  0.0},       # PBC + helix: DMI vanishes by symmetry

    (InterfaceDMIField,      "fem1d"):     {"h_sum":   -925960.875,  "e_sum":    463351.5,     "E":  3.089e-22},
    (InterfaceDMIField,      "fem2d"):     {"h_sum":  -4195535.0,    "e_sum":  -1315146.625,   "E": -5.972e-22},
    (InterfaceDMIField,      "fem3d"):     {"h_sum":   -286506.5,    "e_sum":   -475890.0,     "E": -3.834e-23},
    (InterfaceDMIField,      "fic1d"):     {"h_sum":   -547138.875,  "e_sum":    283505.25,    "E":  2.835e-22},
    (InterfaceDMIField,      "fic2d"):     {"h_sum":    775596.75,   "e_sum":   -233844.16,    "E": -2.338e-22},
    (InterfaceDMIField,      "fic"):       {"h_sum":   -509684.75,   "e_sum":    -64895.14,    "E": -6.49e-23},
    (InterfaceDMIField,      "fem3d_pbc"): {"h_sum":         0.0,    "e_sum":          0.0,    "E":  0.0},       # PBC + helix: interface DMI vanishes by symmetry
    (InterfaceDMIField,      "fic_pbc"):   {"h_sum":         0.0,    "e_sum":          0.0,    "E":  0.0},       # PBC + helix: interface DMI vanishes by symmetry

    (UniaxialAnisotropyField, "fem1d"):    {"h_sum":   3264281.5,    "e_sum":  -1181740.0,     "E": -7.768e-22},
    (UniaxialAnisotropyField, "fem2d"):    {"h_sum":   7096425.0,    "e_sum":  -2833911.0,     "E": -1.32e-21},
    (UniaxialAnisotropyField, "fem3d"):    {"h_sum":   1921947.5,    "e_sum":  -2968750.0,     "E": -9.488e-22},
    (UniaxialAnisotropyField, "fic1d"):    {"h_sum":   1229046.0,    "e_sum":   -254439.6,     "E": -2.544e-22},
    (UniaxialAnisotropyField, "fic2d"):    {"h_sum":   4841155.0,    "e_sum":  -1634743.125,   "E": -1.635e-21},
    (UniaxialAnisotropyField, "fic"):      {"h_sum":   1188822.5,    "e_sum":   -980665.0,     "E": -9.807e-22},
    (UniaxialAnisotropyField, "fem3d_pbc"):{"h_sum":   1188822.5,    "e_sum":   -980665.0,     "E": -9.807e-22},
    (UniaxialAnisotropyField, "fic_pbc"):  {"h_sum":   1188822.5,    "e_sum":    -44635.84,    "E": -4.464e-23},

    (ExternalField,          "fem1d"):     {"h_sum":        18.0,    "e_sum":       -6.653,    "E": -4.559e-27},
    (ExternalField,          "fem2d"):     {"h_sum":        54.0,    "e_sum":       -5.378,    "E": -2.431e-27},
    (ExternalField,          "fem3d"):     {"h_sum":       162.0,    "e_sum":       -2.077,    "E":  2.501e-28},
    (ExternalField,          "fic1d"):     {"h_sum":        12.0,    "e_sum":       -3.840,    "E": -3.840e-27},
    (ExternalField,          "fic2d"):     {"h_sum":        24.0,    "e_sum":       -6.708,    "E": -6.708e-27},
    (ExternalField,          "fic"):       {"h_sum":        48.0,    "e_sum":       -1.313,    "E": -1.313e-27},
    (ExternalField,          "fem3d_pbc"): {"h_sum":        48.0,    "e_sum":       -1.313,    "E": -1.313e-27},
    (ExternalField,          "fic_pbc"):   {"h_sum":        48.0,    "e_sum":       -1.313,    "E": -1.313e-27},
    # fmt: on
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

    if field_cls is ExternalField:
        if set(state.m.spaces) == {"c"}:
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

    # For physically-zero references, only use absolute tolerance (sized to the
    # noise floor); tiny nonzero roundoff values are not reproducible across
    # CPU/GPU/precision due to differing reduction orders.
    _close = lambda actual, ref_v, atol_zero, rel, atol: (
        actual == pytest.approx(0.0, abs=atol_zero)
        if ref_v == 0.0
        else actual == pytest.approx(ref_v, rel=rel, abs=atol)
    )
    assert _close(h_sum, ref["h_sum"], atol_zero=100.0, rel=1e-3, atol=20.0)
    assert _close(e_sum, ref["e_sum"], atol_zero=10.0, rel=1e-3, atol=1.0)
    assert _close(E, ref["E"], atol_zero=1e-25, rel=1e-2, atol=1e-28)
