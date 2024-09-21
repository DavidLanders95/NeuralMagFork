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

import importlib
import os

from neuralmag.common import logging

__all__ = ["config"]


class Config:
    def __init__(self):
        self._backend = None
        self._device = None
        self._dtype = None

        # public config keys
        self.torch = {"compile": True}
        self.jax = {"jit": True}
        self.fem = {"n_gauss": 3}

    @property
    def backend(self):
        if self._backend is not None:
            return self._backend

        backend_name = os.getenv("NM_BACKEND", None)
        if backend_name is not None:
            self.backend = backend_name
        else:
            self.set_backend("jax", True)
            if self._backend is None:
                self.set_backend("torch", True)
            if self._backend is None:
                raise ImportError(
                    "Neither 'jax' nor 'torch' seems to be available. Use 'pip install neuralmag[torch]' or 'pip install neuralmag[jax]' to select one backend!"
                )

        return self._backend

    def set_backend(self, backend_name, graceful=False):
        if self._backend is not None:
            raise RuntimeError(
                "Backend is already set. Cannot switch backend at runtime."
            )
        if backend_name not in ["torch", "jax"]:
            raise ValueError(f"Unsupported backend: {backend_name}")

        self.backend_name = backend_name

        try:
            if backend_name == "torch":
                self._backend = importlib.import_module("neuralmag.backends.torch")
            if backend_name == "jax":
                self._backend = importlib.import_module("neuralmag.backends.jax")
            logging.info_green(f"[NeuralMag] Backend set to '{backend_name}'.")
        except ImportError:
            if not graceful:
                raise ImportError(
                    f"[NeuralMag] Could not load backend '{backend_name}'. Use 'pip install neuralmag[{backend_name}] to install."
                )

    @backend.setter
    def backend(self, backend_name):
        self.set_backend(backend_name)

    @property
    def device(self):
        if self._device is None:
            self.device = self.backend.default_device_str()
        return self._device

    @device.setter
    def device(self, device):
        self._device = self.backend.device_from_str(device)
        logging.info_green(f"[NeuralMag] Set default device to '{device}'.")

    @property
    def dtype(self):
        if self._dtype is None:
            self.dtype = self.backend.default_dtype_str()
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = self.backend.dtype_from_str(dtype)
        logging.info_green(f"[NeuralMag] Set default dtype to '{dtype}'.")


config = Config()
