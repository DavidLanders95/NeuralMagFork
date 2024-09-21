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

import jax
import jax.numpy as jnp
import numpy as np

from neuralmag.common import config, logging

float64 = jnp.float64
float32 = jnp.float32
integer = jnp.int32
Tensor = jax.Array

np = jnp
libs = {"jnp": jnp}


def device_from_str(device):
    return None


def device_for_state(device):
    logging.warning(
        f"[NeuralMag] JAX backend doesn't support setting device per State. Falling back to default device."
    )
    return None


def default_device_str():
    return jax.devices()[0]


def dtype_from_str(dtype):
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    return {"float64": float64, "float32": float32}[dtype]


def dtype_for_state(dtype):
    logging.warning(
        f"[NeuralMag] JAX backend doesn't support setting dtype per State. Falling back to default dtype."
    )
    return config.dtype


def default_dtype_str():
    return "float32"


def eps(dtype):
    return jnp.finfo(dtype).eps


def tensor(value, *, device=None, dtype=None):
    # if isinstance(value, jax.Array):
    #    if value.device != device:
    #        return jax.device_put(value, device)
    #    else:
    #        return value
    return jnp.array(value, dtype=dtype)


def zeros(shape, *, device=None, dtype=None, **kwargs):
    return jnp.zeros(shape, dtype=dtype, **kwargs)


def zeros_like(tensor):
    return jnp.zeros_like(tensor)


def arange(*args, device=None, dtype=None, **kwargs):
    return jnp.arange(*args, dtype=dtype, **kwargs)


def linspace(*args, device=None, dtype=None, **kwargs):
    return jnp.linspace(*args, dtype=dtype, **kwargs)


def meshgrid(*ranges, indexing="ij"):
    return jnp.meshgrid(*ranges, indexing="ij")


def to_numpy(array):
    return np.asarray(array)


def broadcast_to(array, shape):
    return jnp.broadcast_to(array, shape)


def tile(array, shape):
    return jnp.tile(array, shape)


def assign(target, source, idx):
    return target.at[idx].set(source)


def mean(tensor, axis=None):
    return jnp.mean(tensor, axis=axis)
