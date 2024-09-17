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

__all__ = ["config"]


class Config:
    def __init__(self):
        self._backend = None

        # public config keys
        self.torch = {"compile": True}
        self.fem = {"n_gauss": 3}

    @property
    def backend(self):
        if self._backend is None:
            try:  # check environment variable first
                self.backend = os.environ["BACKEND"]
            except:
                try:  # then try torch by default
                    import torch

                    self.backend = "torch"
                except ImportError:
                    try:  # finally try jax
                        import jax

                        self.backend = "jax"
                    except ImportError:
                        pass

        if self._backend is None:
            raise ImportError("Neither 'jax' nor 'torch' seems to be available")
        return self._backend

    @backend.setter
    def backend(self, backend_name):
        # TODO raise error if backend is already set?
        if backend_name not in ["torch", "jax"]:
            raise ValueError(f"Unsupported backend: {backend_name}")

        self.backend_name = backend_name
        if backend_name == "torch":
            self._backend = importlib.import_module("neuralmag.backends.torch")
        elif backend_name == "jax":
            self._backend = importlib.import_module("neuralmag.backends.jax")


config = Config()
