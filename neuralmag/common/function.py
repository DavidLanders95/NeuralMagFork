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

__all__ = ["Function", "VectorFunction", "CellFunction", "VectorCellFunction"]


class Function(object):
    """
    This class represents a discretized field on the mesh of a state object.

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

        size = []
        for i, space in enumerate(spaces):
            if space == "c":
                size.append(state.mesh.n[i])
            elif space == "n":
                size.append(state.mesh.n[i] + 1)
            else:
                raise Exception(f"Function space '{space}' not supported")

        self._size = tuple(size) + shape

        if tensor is None or isinstance(tensor, torch.Tensor):
            self._tensor = tensor
        else:
            raise NotImplemented("Unsupported tensor type.")
        self._expanded = False

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
    def size(self):
        """
        The size (shape) of the tensor with the discretized field values
        """
        return self._size

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
                self._size, dtype=self._state.dtype, device=self._state.device
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
        the constant to the full mesh size.

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
                ).expand(self._size)
            elif isinstance(constant, (list, tuple)):
                assert self.shape == (3,)
                self._tensor = self._expanded.reshape(
                    (1,) * self.state.mesh.dim + (3,)
                ).expand(self._size)
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
        # TODO get rid of conditionals?
        # TODO support all function spaces
        if self._spaces == "ccc" or self._spaces == "cc":
            return self.tensor.mean(dim=tuple(range(self._state.mesh.dim)))
        if self._spaces == "nn":
            return (
                (
                    self.tensor[1:, 1:, ...]
                    + self.tensor[:-1, 1:, ...]
                    + self.tensor[1:, :-1, ...]
                    + self.tensor[:-1, :-1, ...]
                )
                / 4.0
            ).mean(dim=(0, 1))
        elif self._spaces == "nnn":
            return (
                (
                    self.tensor[1:, 1:, 1:, ...]
                    + self.tensor[:-1, 1:, 1:, ...]
                    + self.tensor[1:, :-1, 1:, ...]
                    + self.tensor[:-1, :-1, 1:, ...]
                    + self.tensor[1:, 1:, :-1, ...]
                    + self.tensor[:-1, 1:, :-1, ...]
                    + self.tensor[1:, :-1, :-1, ...]
                    + self.tensor[:-1, :-1, :-1, ...]
                )
                / 8.0
            ).mean(dim=(0, 1, 2))
        else:
            raise Exception(f"Function space '{self._spaces}' not supported by mean.")


class CellFunction(Function):
    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        super().__init__(state, **kwargs)


class VectorFunction(Function):
    def __init__(self, state, **kwargs):
        assert "shape" not in kwargs
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)


class VectorCellFunction(Function):
    def __init__(self, state, **kwargs):
        assert "spaces" not in kwargs
        assert "shape" not in kwargs
        kwargs["spaces"] = "c" * state.mesh.dim
        kwargs["shape"] = (3,)
        super().__init__(state, **kwargs)
