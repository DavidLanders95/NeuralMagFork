"""
NeuralMag - A nodal finite-difference code for inverse micromagnetics

Copyright (c) 2024 NeuralMag team

This program is free software: you can redistribute it and/or modify
it under the terms of the Lesser Python General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
Lesser Python General Public License for more details.

You should have received a copy of the Lesser Python General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch

from ..generators import pytorch_generator as gen

__all__ = ["Function", "VectorFunction", "CellFunction", "VectorCellFunction"]


class Function(gen.CodeClass):
    """
    This class represents a discretized field on the mesh of a state object.

    If the instance is not intialized with a tensor, the tensor is lazy-
    initialized with zeros on the first access.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param spaces: The function spaces in the principal direction (defaults to
        "nnn" for 3D meshes and "nn" for 2D meshes)
    :type spaces: str
    :param shape: The shape of the function (either ``()`` for scalars or ``(3,)`` for
        vectors is currently supported)
    :type shape: tuple
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, spaces=None, shape=(), tensor=None, name=None):
        self._state = state
        if spaces is None:
            spaces = "n" * state.mesh.dim
        self._spaces = spaces
        self._shape = shape
        if name is None:
            self._name = "f"
        else:
            self._name = name

        tensor_shape = []
        for i, space in enumerate(spaces):
            if space == "c":
                tensor_shape.append(state.mesh.n[i])
            elif space == "n":
                tensor_shape.append(state.mesh.n[i] + 1)
            else:
                raise Exception(f"Function space '{space}' not supported")

        self._tensor_shape = tuple(tensor_shape) + shape

        if tensor is None or isinstance(tensor, torch.Tensor):
            self._tensor = tensor
        else:
            raise NotImplemented("Unsupported tensor type.")
        self._expanded = False

        self.save_and_load_code(spaces, shape)

    @property
    def name(self):
        """
        The name of the function
        """
        return self._name

    @property
    def shape(self):
        """
        The shape of the function
        """
        return self._shape

    @property
    def tensor_shape(self):
        """
        The shape of the tensor with the discretized field values
        """
        return self._tensor_shape

    @property
    def state(self):
        """
        The state object used for the construction of the function
        """
        return self._state

    @property
    def spaces(self):
        """
        The function spaces of the function
        """
        return self._spaces

    @property
    def tensor(self):
        """
        The tensor containing the discretized values of the function
        """
        if self._tensor is None:
            self._tensor = torch.zeros(
                self._tensor_shape, dtype=self._state.dtype, device=self._state.device
            )
        return self._tensor

    def fill(self, constant, expand=False):
        """
        Fills the tensor of the function with a constant value.

        :param constant: The constant to fill the tensor
        :type constant: int, list
        :param expand: If True, the tensor is set by expanding the constant
            to the size of the mesh using :code:`torch.Tensor.expand`
            resulting in minimal storage consumption.
        :type expand: bool
        :return: The function itself
        :rvalue: :class:`Function`

        :Example:
            .. code-block::

                state = nm.State(nm.Mesh((10, 10, 10), (1e-9, 1e-9, 1e-9)))
                f = nm.Function(state, shape = (3,)).fill([1.0, 2.0, 3.0])
        """
        if expand:
            return self.fill_expanded(constant)
        if isinstance(constant, (int, float)):
            assert self.shape == ()
            self.tensor[...] = constant
        elif isinstance(constant, (list, tuple)):
            assert self.shape == (3,)
            for i in range(3):
                self.tensor[..., i] = constant[i]
        else:
            raise NotImplemented("Unsupported shape.")

        return self

    def fill_expanded(self, constant):
        """
        Fills the tensor of the function with a constant value by expanding
        the constant to the full mesh size by to use of :code:`torch.Tensor.expand`.

        This reduces the memory consumption of the tensor to a single double value.

        :param constant: The constant to fill the tensor
        :type constant: int, list
        :return: The function itself
        :rvalue: :class:`Function`
        """
        if self._tensor is None:
            self._expanded = self.state.tensor(constant)
            if isinstance(constant, (int, float)):
                assert self.shape == ()
                self._tensor = self._expanded.reshape(
                    (1,) * self.state.mesh.dim
                ).expand(self._tensor_shape)
            elif isinstance(constant, (list, tuple)):
                assert self.shape == (3,)
                self._tensor = self._expanded.reshape(
                    (1,) * self.state.mesh.dim + (3,)
                ).expand(self._tensor_shape)
            else:
                raise NotImplemented("Unsupported shape.")
        elif self._expanded is not None:
            self._expanded[:] = self.state.tensor(constant)
        else:
            raise Exception(
                "Cannot transform a regular Function to an expanded Function"
            )

        return self

    def avg(self):
        """
        Returns the componentwise average of the function over the mesh.

        :return: The componentwise average
        :rtype: :class:`torch.Tensor`
        """
        return self._code.avg(self._state.rho.tensor, self._state.dx, self.tensor)

    @classmethod
    def _generate_code(cls, spaces, shape):
        code = gen.CodeBlock()
        dim = len(spaces)

        # generate avg method
        f = gen.Variable("f", spaces, shape)
        with code.add_function("avg", ["rho", "dx", "f"]) as func:
            terms, _ = gen.compile_functional(1 * gen.dV(dim))
            func.assign_sum("vol", *[term["cmd"] for term in terms])

            if shape == ():
                terms, variables = gen.compile_functional(f * gen.dV(dim))
                func.assign_sum("fint", *[term["cmd"] for term in terms])
            elif shape == (3,):
                func.zeros_like("fint", "f", (3,))
                for i in range(3):
                    terms, _ = gen.compile_functional(f.dot(gen.cs_e[i]) * gen.dV(dim))
                    func.assign_sum(f"fint[{i}]", *[term["cmd"] for term in terms])

            func.retrn("fint / vol")

        return code


class CellFunction(Function):
    """
    Subclass of :class:`Function` with the function space set to cellwise in each dimension.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :type spaces: str
    :param shape: The shape of the function (either ``()`` for scalars or ``(3,)`` for
        vectors is currently supported)
    :type shape: tuple
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        super().__init__(state, **kwargs)


class VectorFunction(Function):
    """
    Subclass of :class:`Function` with the shape set to (3,).

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param spaces: The function spaces in the principal direction (defaults to
        "nnn" for 3D meshes and "nn" for 2D meshes)
    :type spaces: str
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "shape" not in kwargs
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)


class VectorCellFunction(Function):
    """
    Subclass of :class:`Function` with the shape set to (3,) and the
    function space set to cellwise in each dimension.

    :param state: The state that is used to construct the function
    :type state: :class:`State`
    :param tensor: Tensor with discretized function values
    :type tensor: :class:`torch.Tensor`
    :param name: Name of the function
    :type name: str
    """

    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        assert "shape" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)
