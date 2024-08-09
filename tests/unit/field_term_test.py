import pytest

from neuralmag import *


def test_subclassing(state):
    class SomeField(FieldTerm):
        default_name = "some_field"

    with pytest.raises(TypeError):

        class SomeFieldWithoutDefaultname(FieldTerm):
            pass
