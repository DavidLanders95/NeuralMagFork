import torch

__all__ = ["Function", "VectorFunction", "CellFunction", "VectorCellFunction"]


class Function(object):
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
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    @property
    def state(self):
        return self._state

    @property
    def spaces(self):
        return self._spaces

    @property
    def tensor(self):
        if self._tensor is None:
            self._tensor = torch.zeros(
                self._size, dtype=self._state.dtype, device=self._state.device
            )
        return self._tensor

    def fill(self, constant, expand=False):
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
