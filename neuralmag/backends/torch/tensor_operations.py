# SPDX-License-Identifier: MIT

import os

import torch

from neuralmag.common import logging

float64 = torch.float64
float32 = torch.float32
integer = torch.int
Tensor = torch.Tensor

np = torch
libs = {"torch": torch}


def device_from_str(device):
    return torch.device(device)


def device_for_state(device):
    return device_from_str(device)


def default_device_str():
    return (
        f"cuda:{os.environ.get('CUDA_DEVICE', '0')}"
        if torch.cuda.is_available()
        else "cpu"
    )


def dtype_from_str(dtype):
    return {"float64": float64, "float32": float32}[dtype]


def dtype_for_state(dtype):
    return dtype_from_str(dtype)


def default_dtype_str():
    return "float32"


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


def zeros_like(tensor):
    return torch.zeros_like(tensor)


def arange(*args, device=None, dtype=None, **kwargs):
    return torch.arange(*args, device=device, dtype=dtype, **kwargs)


def linspace(*args, device=None, dtype=None, **kwargs):
    return torch.linspace(*args, device=device, dtype=dtype, **kwargs)


def meshgrid(*ranges, indexing="ij"):
    return torch.meshgrid(*ranges, indexing="ij")


def to_numpy(array):
    return array.detach().cpu().numpy()


def broadcast_to(array, shape):
    return array.expand(shape)


def tile(array, shape):
    return torch.tile(array, shape)


def assign(target, source, idx):
    target[idx] = source
    return target


def mean(tensor, axis=None):
    return torch.mean(tensor, dim=axis)
