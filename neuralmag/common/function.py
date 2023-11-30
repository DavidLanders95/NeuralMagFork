import torch

__all__ = ["Function", "VectorFunction", "CellFunction", "VectorCellFunction"]


class Function(object):
    def __init__(self, state, ftype="node", shape=(), tensor=None, name=None):
        self._state = state
        self._ftype = ftype
        self._shape = shape
        if name is None:
            self._name = "f"
        else:
            self._name = name

        if ftype == "node":
            self._size = tuple([n + 1 for n in state.mesh.n])
        elif ftype == "cell":
            self._size = state.mesh.n
        else:
            raise NotImplemented(f'Unknown ftype "{ftype}".')

        self._size += shape

        if tensor is None:
            self._tensor = torch.zeros(
                self._size, dtype=state.dtype, device=state.device
            )
        elif isinstance(tensor, torch.Tensor):
            self._tensor = tensor
        else:
            raise NotImplemented("Unsupported tensor type.")

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def state(self):
        return self._state

    @property
    def ftype(self):
        return self._ftype

    @property
    def tensor(self):
        return self._tensor

    def from_constant(self, constant):
        if isinstance(constant, (int, float)):
            assert self._shape == ()
            self._tensor[...] = constant
        elif isinstance(constant, (list, tuple)):
            assert self._shape == (3,)
            for i in range(3):
                self._tensor[..., i] = constant[i]
        else:
            raise NotImplemented("Unsupported shape.")

        return self

    def avg(self):
        # TODO get rid of conditionals?
        if self._ftype == "cell":
            return self._tensor.mean(dim=tuple(range(self._state.mesh.dim)))
        else:
            if self._state.mesh.dim == 2:
                return (
                    (
                        +self._tensor[1:, 1:, ...]
                        + self._tensor[:-1, 1:, ...]
                        + self._tensor[1:, :-1, ...]
                        + self._tensor[:-1, :-1, ...]
                    )
                    / 4.0
                ).mean(dim=(0, 1))
            elif self._state.mesh.dim == 3:
                return (
                    (
                        +self._tensor[1:, 1:, 1:, ...]
                        + self._tensor[:-1, 1:, 1:, ...]
                        + self._tensor[1:, :-1, 1:, ...]
                        + self._tensor[:-1, :-1, 1:, ...]
                        + self._tensor[1:, 1:, :-1, ...]
                        + self._tensor[:-1, 1:, :-1, ...]
                        + self._tensor[1:, :-1, :-1, ...]
                        + self._tensor[:-1, :-1, :-1, ...]
                    )
                    / 8.0
                ).mean(dim=(0, 1, 2))
            else:
                raise


class CellFunction(Function):
    def __init__(self, *args, **kwargs):
        assert "ftype" not in kwargs
        kwargs["ftype"] = "cell"
        super().__init__(*args, **kwargs)


class VectorFunction(Function):
    def __init__(self, *args, **kwargs):
        assert "shape" not in kwargs
        kwargs["shape"] = (3,)
        super().__init__(*args, **kwargs)


class VectorCellFunction(Function):
    def __init__(self, *args, **kwargs):
        assert "ftype" not in kwargs
        assert "shape" not in kwargs
        kwargs["ftype"] = "cell"
        kwargs["shape"] = (3,)
        super().__init__(*args, **kwargs)
