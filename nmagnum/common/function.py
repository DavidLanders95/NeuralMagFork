import os

import numpy as np
import pyvista as pv
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

    def from_numpy(self, array):
        assert array.shape == self._size
        self._tensor = torch.tensor(
            data=array, dtype=self._state.dtype, device=self._state.device
        )
        return self

    def avg(self):
        return self._tensor.mean(dim=(0, 1, 2))

    # def normalize(self):
    #    self /= torch.linalg.norm(self, dim = -1, keepdim = True)
    #    self[...] = torch.nan_to_num(self, posinf=0, neginf=0)
    #    return self

    def write(self, filename):
        n = self._state.mesh.n
        dx = self._state.mesh.dx
        origin = self._state.mesh.origin

        grid = pv.UniformGrid(dimensions=np.array(n) + 1, spacing=dx, origin=origin)

        if self.shape == ():
            data = self.tensor.detach().cpu().numpy().flatten("F")
        elif self.shape == (3,):
            data = self.tensor.detach().cpu().numpy().reshape(-1, 3, order="F")
        else:
            raise NotImplemented("Unsupported shape.")

        if self._ftype == "node":
            grid.point_data.set_array(data, self._name)
        elif self._ftype == "cell":
            grid.cell_data.set_array(data, self._name)
        else:
            raise NotImplemented("Unsupported ftype.")

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        grid.save(filename)


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
