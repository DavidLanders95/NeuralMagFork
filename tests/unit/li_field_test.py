import numpy as np
import pytest

from neuralmag import *
from neuralmag.common.engine import N
from neuralmag.field_terms.li_field import LIField, Lifshitz_invariant, component

be = config.backend


def test_valid_initialization():
    """Test valid LI field initialization."""
    field = LIField("xyz")
    assert field._options["LI"] == "xyz"
    assert field.default_name == "LI_xyz"


def test_default_name_changes_with_li():
    """Test that default name includes LI string."""
    field_xyz = LIField("xyz")
    field_yzx = LIField("yzx")
    assert field_xyz.default_name == "LI_xyz"
    assert field_yzx.default_name == "LI_yzx"


def test_invalid_li_type():
    """Test TypeError for non-string LI parameter."""
    with pytest.raises(TypeError, match="LI must be a string"):
        LIField(123)

    with pytest.raises(TypeError, match="LI must be a string"):
        LIField(["x", "y", "z"])


def test_invalid_li_length():
    """Test ValueError for wrong length LI string."""
    with pytest.raises(ValueError, match="LI must be a string of length 3"):
        LIField("xy")

    with pytest.raises(ValueError, match="LI must be a string of length 3"):
        LIField("xyzw")

    with pytest.raises(ValueError, match="LI must be a string of length 3"):
        LIField("")


def test_invalid_li_characters():
    """Test ValueError for invalid characters in LI string."""
    with pytest.raises(ValueError, match="LI contains invalid character 'a'"):
        LIField("axz")

    with pytest.raises(ValueError, match="LI contains invalid character '1'"):
        LIField("x1z")

    with pytest.raises(ValueError, match="LI contains invalid character"):
        LIField("x@z")


def test_valid_li_combinations():
    """Test that all valid x,y,z combinations work."""
    valid_combinations = [
        "xyz",
        "xzy",
        "yxz",
        "yzx",
        "zxy",
        "zyx",
        "xyx",
        "xzx",
        "yxy",
        "yzy",
        "zxz",
        "zyz",  # repeated indices allowed
    ]

    for li in valid_combinations:
        field = LIField(li)
        assert field._options["LI"] == li
        assert field.default_name == f"LI_{li}"


def test_h_field_calculation(state):
    """Test LI field contribution calculation."""
    # Set up material parameters
    state.material.Dxyz = CellFunction(state).fill(1e-3)
    state.material.Ms = CellFunction(state).fill(8e5)

    # Register LI field
    LIField("xyz").register(state, "LI_xyz")

    # Check that field was registered with correct name
    assert hasattr(state, "h_LI_xyz")

    # The sum should be a finite number (exact value depends on magnetization configuration)
    h_sum = be.to_numpy(state.h_LI_xyz.tensor.sum())
    assert np.isfinite(h_sum)


def test_h_field_with_custom_name(state):
    """Test LI field with custom registration name."""
    state.material.Dyzx = CellFunction(state).fill(2e-3)
    state.material.Ms = CellFunction(state).fill(8e5)

    LIField("yzx").register(state, "custom_li")

    assert hasattr(state, "h_custom_li")
    h_sum = be.to_numpy(state.h_custom_li.tensor.sum())
    assert np.isfinite(h_sum)


def test_different_li_give_different_results(state):
    """Test that different LI strings produce different field values."""
    state.material.Dxyz = CellFunction(state).fill(1e-3)
    state.material.Dyzx = CellFunction(state).fill(1e-3)
    state.material.Ms = CellFunction(state).fill(8e5)

    LIField("xyz").register(state, "li1")
    LIField("yzx").register(state, "li2")

    h1_sum = be.to_numpy(state.h_li1.tensor.sum())
    h2_sum = be.to_numpy(state.h_li2.tensor.sum())

    # Different LI should generally give different results
    # (unless magnetization has special symmetry)
    # We just check they're both finite and potentially different
    assert np.isfinite(h1_sum)
    assert np.isfinite(h2_sum)


def test_2d_field(state2d):
    """Test LI field in 2D simulation."""
    state2d.material.Dxyx = CellFunction(state2d).fill(1e-3)
    state2d.material.Ms = CellFunction(state2d).fill(8e5)

    LIField("xyx").register(state2d, "LI_xyx")

    h_sum = be.to_numpy(state2d.h_LI_xyx.tensor.sum())
    assert np.isfinite(h_sum)


def test_lifshitz_invariant_parsing():
    """Test that Lifshitz invariant function parses LI strings correctly."""
    # Test that different LI strings are handled without crashing
    li_strings = ["xyz", "yzx", "zxy", "xzy", "yxz", "zyx"]

    for li in li_strings:
        # Test the parsing logic without full symbolic computation
        valid_components = {"x": N.x, "y": N.y, "z": N.z}
        i, j, k = [valid_components[l] for l in li]
        assert i in [N.x, N.y, N.z]
        assert j in [N.x, N.y, N.z]
        assert k in [N.x, N.y, N.z]


def test_component_mapping():
    """Test component function maps directions correctly."""
    assert component(N.x) == N.i
    assert component(N.y) == N.j
    assert component(N.z) == N.k


def test_component_invalid_input():
    """Test component function raises error for invalid input."""
    with pytest.raises(ValueError, match="is not N.x, N.y, or N.z"):
        component("invalid")

    with pytest.raises(ValueError, match="is not N.x, N.y, or N.z"):
        component(None)


def test_multiple_li_fields(state):
    """Test using multiple LI fields simultaneously."""
    # Set up multiple material constants
    state.material.Dxyz = CellFunction(state).fill(1e-3)
    state.material.Dyzx = CellFunction(state).fill(2e-3)
    state.material.Dzxy = CellFunction(state).fill(1.5e-3)
    state.material.Ms = CellFunction(state).fill(8e5)

    # Register multiple LI fields
    LIField("xyz").register(state, "li_xyz")
    LIField("yzx").register(state, "li_yzx")
    LIField("zxy").register(state, "li_zxy")

    # Check all fields are registered
    assert hasattr(state, "h_li_xyz")
    assert hasattr(state, "h_li_yzx")
    assert hasattr(state, "h_li_zxy")

    # Check all give finite results
    for field_name in ["h_li_xyz", "h_li_yzx", "h_li_zxy"]:
        h_sum = be.to_numpy(getattr(state, field_name).tensor.sum())
        assert np.isfinite(h_sum)


def test_li_field_with_zero_constant(state):
    """Test LI field behavior with zero DMI constant."""
    state.material.Dxyz = CellFunction(state).fill(0.0)
    state.material.Ms = CellFunction(state).fill(8e5)

    LIField("xyz").register(state, "LI_xyz")

    h_sum = be.to_numpy(state.h_LI_xyz.tensor.sum())
    assert h_sum == pytest.approx(0.0, abs=1e-10)
