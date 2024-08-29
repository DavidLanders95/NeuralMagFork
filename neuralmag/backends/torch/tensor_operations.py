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

float64 = torch.float64
Tensor = torch.Tensor


def device(device):
    return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")


def eps(dtype):
    return torch.finfo(dtype).eps


def tensor(value, *, device=None, dtype=None, requires_grad=False):
    if isinstance(value, torch.Tensor):
        if value.device != device:
            return value.to(device)
        else:
            return value
    return torch.tensor(value, device=device, dtype=dtype, requires_grad=requires_grad)


def zeros(shape, *, device=None, dtype=None, **kwargs):
    return torch.zeros(shape, device=device, dtype=dtype, **kwargs)


def arange(*args, device=None, dtype=None, **kwargs):
    return torch.arange(*args, device=device, dtype=dtype, **kwargs)


def meshgrid(*ranges, indexing="ij"):
    return torch.meshgrid(*ranges, indexing="ij")


def to_numpy(array):
    return array.detach().cpu().numpy()


def broadcast_to(array, shape):
    return array.expand(shape)


def tile(array, shape):
    return torch.tile(array, shape)
