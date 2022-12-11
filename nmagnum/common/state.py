import os

import numpy as np
import torch

from . import logging

__all__ = ["State"]


class Material:
    pass


class State(object):
    def __init__(self, mesh, t0=0.0, device=None):
        self.mesh = mesh
        if device == None:
            CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "0")
            self._device = torch.device(
                f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = device

        self.material = Material()

        self.t = t0
        logging.info_green("[State] running on device:%s" % self._device)
        logging.info_green("[Mesh] %dx%dx%d (size= %g x %g x %g)" % (mesh.n + mesh.dx))

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return torch.float64

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if isinstance(value, torch.Tensor):
            self._t = value
        else:
            self._t = torch.tensor(value, dtype=self.dtype, device=self.device)
