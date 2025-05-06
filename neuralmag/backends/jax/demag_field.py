# SPDX-License-Identifier: MIT

import os
from time import time

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
from jax import grad, vmap
from jax.numpy import abs
from jax.numpy import arcsinh as asinh
from jax.numpy import arctan as atan
from jax.numpy import log, pi, sqrt
from scipy import constants

from neuralmag.common import logging
from neuralmag.common.engine import Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

complex_dtype = {
    jnp.dtype("float32"): jnp.complex64,
    jnp.dtype("float64"): jnp.complex128,
}


def f(x, y, z):
    x, y, z = abs(x), abs(y), abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = sqrt(x2 + y2 + z2)
    res = 1.0 / 6.0 * (2 * x2 - y2 - z2) * r
    res += jnp.nan_to_num(
        (y / 2.0 * (z2 - x2) * asinh(y / sqrt(x2 + z2))), posinf=0, neginf=0
    )
    res += jnp.nan_to_num(
        (z / 2.0 * (y2 - x2) * asinh(z / sqrt(x2 + y2))), posinf=0, neginf=0
    )
    res -= jnp.nan_to_num((x * y * z * atan(y * z / (x * r))), posinf=0, neginf=0)
    return res


def g(x, y, z):
    z = abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = sqrt(x2 + y2 + z2)
    res = -x * y * r / 3.0
    res += jnp.nan_to_num((x * y * z * asinh(z / sqrt(x2 + y2))), posinf=0, neginf=0)
    res += jnp.nan_to_num(
        (y / 6.0 * (3.0 * z2 - y2) * asinh(x / sqrt(y2 + z2))), posinf=0, neginf=0
    )
    res += jnp.nan_to_num(
        (x / 6.0 * (3.0 * z2 - x2) * asinh(y / sqrt(x2 + z2))), posinf=0, neginf=0
    )
    res -= jnp.nan_to_num((z**3 / 6.0 * atan(x * y / (z * r))), posinf=0, neginf=0)
    res -= jnp.nan_to_num((z * y2 / 2.0 * atan(x * z / (y * r))), posinf=0, neginf=0)
    res -= jnp.nan_to_num((z * x2 / 2.0 * atan(y * z / (x * r))), posinf=0, neginf=0)
    return res


def F1(func, x, y, z, dz, dZ):
    return (
        func(x, y, z + dZ)
        - func(x, y, z)
        - func(x, y, z - dz + dZ)
        + func(x, y, z - dz)
    )


def F0(func, x, y, z, dy, dY, dz, dZ):
    return (
        F1(func, x, y + dY, z, dz, dZ)
        - F1(func, x, y, z, dz, dZ)
        - F1(func, x, y - dy + dY, z, dz, dZ)
        + F1(func, x, y - dy, z, dz, dZ)
    )


def newell(func, x, y, z, dx, dy, dz, dX, dY, dZ):
    res = (
        F0(func, x, y, z, dy, dY, dz, dZ)
        - F0(func, x - dx, y, z, dy, dY, dz, dZ)
        - F0(func, x + dX, y, z, dy, dY, dz, dZ)
        + F0(func, x - dx + dX, y, z, dy, dY, dz, dZ)
    )
    return -res / (4.0 * pi * dx * dy * dz)


def dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    res = (2.0 * x**2 - y**2 - z**2) * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    res = res.at[0, 0, 0].set(0.0)
    return res * dx * dy * dz / (4.0 * pi)


def dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    res = 3.0 * x * y * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    res = res.at[0, 0, 0].set(0.0)
    return res * dx * dy * dz / (4.0 * pi)


def demag_f(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(
        dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2
    ) < p**2
    res = res.at[near].set(newell(f, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ))
    return res


def demag_g(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(
        dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2
    ) < p**2
    res = res.at[near].set(newell(g, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ))
    return res


def h_cell(N_demag, mcell, material__Ms, rho):
    dim = [i for i in range(3) if mcell.shape[i] > 1]
    s = [mcell.shape[i] * 2 for i in dim]

    hx = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[mcell.dtype])
    hy = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[mcell.dtype])
    hz = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[mcell.dtype])

    for ax in range(3):
        m_pad_fft1D = jnp.fft.rfftn(
            rho * material__Ms * mcell[:, :, :, ax], axes=dim, s=s
        )
        hx += N_demag[0][ax] * m_pad_fft1D
        hy += N_demag[1][ax] * m_pad_fft1D
        hz += N_demag[2][ax] * m_pad_fft1D

    hx = jnp.fft.irfftn(hx, axes=dim)
    hy = jnp.fft.irfftn(hy, axes=dim)
    hz = jnp.fft.irfftn(hz, axes=dim)

    return jnp.stack(
        [
            hx[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
            hy[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
            hz[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
        ],
        axis=3,
    )


@jax.jit
def h2d(N_demag, m, material__Ms, rho):
    mcell = (
        jnp.expand_dims(
            m[1:, 1:, :] + m[:-1, 1:, :] + m[1:, :-1, :] + m[:-1, :-1, :], -2
        )
        / 4.0
    )

    # TODO behavior for unsqueezed scalars seems OK, but not so elegant?
    Ms_hcell = jnp.expand_dims(rho * material__Ms, -1) * h_cell(
        N_demag, mcell, jnp.expand_dims(material__Ms, -1), jnp.expand_dims(rho, -1)
    ).squeeze(-2)

    h = jnp.zeros(m.shape, dtype=m.dtype)
    h = h.at[:-1, :-1].add(Ms_hcell)
    h = h.at[:-1, 1:].add(Ms_hcell)
    h = h.at[1:, :-1].add(Ms_hcell)
    h = h.at[1:, 1:].add(Ms_hcell)

    mass = jnp.zeros(h.shape[:-1], dtype=h.dtype)
    mass = mass.at[:-1, :-1].add(rho * material__Ms)
    mass = mass.at[:-1, 1:].add(rho * material__Ms)
    mass = mass.at[1:, :-1].add(rho * material__Ms)
    mass = mass.at[1:, 1:].add(rho * material__Ms)

    return h / jnp.expand_dims(mass, -1)


@jax.jit
def h3d(N_demag, m, material__Ms, rho):
    mcell = (
        +m[1:, 1:, 1:, :]
        + m[:-1, 1:, 1:, :]
        + m[1:, :-1, 1:, :]
        + m[:-1, :-1, 1:, :]
        + m[1:, 1:, :-1, :]
        + m[:-1, 1:, :-1, :]
        + m[1:, :-1, :-1, :]
        + m[:-1, :-1, :-1, :]
    ) / 8.0

    Ms_hcell = jnp.expand_dims(rho * material__Ms, -1) * h_cell(
        N_demag, mcell, material__Ms, rho
    )

    h = jnp.zeros(m.shape, dtype=m.dtype)
    h = h.at[:-1, :-1, :-1].add(Ms_hcell)
    h = h.at[:-1, :-1, 1:].add(Ms_hcell)
    h = h.at[:-1, 1:, :-1].add(Ms_hcell)
    h = h.at[:-1, 1:, 1:].add(Ms_hcell)
    h = h.at[1:, :-1, :-1].add(Ms_hcell)
    h = h.at[1:, :-1, 1:].add(Ms_hcell)
    h = h.at[1:, 1:, :-1].add(Ms_hcell)
    h = h.at[1:, 1:, 1:].add(Ms_hcell)

    mass = jnp.zeros(h.shape[:-1], dtype=h.dtype)
    mass = mass.at[:-1, :-1, :-1].add(rho * material__Ms)
    mass = mass.at[:-1, :-1, 1:].add(rho * material__Ms)
    mass = mass.at[:-1, 1:, :-1].add(rho * material__Ms)
    mass = mass.at[:-1, 1:, 1:].add(rho * material__Ms)
    mass = mass.at[1:, :-1, :-1].add(rho * material__Ms)
    mass = mass.at[1:, :-1, 1:].add(rho * material__Ms)
    mass = mass.at[1:, 1:, :-1].add(rho * material__Ms)
    mass = mass.at[1:, 1:, 1:].add(rho * material__Ms)

    return h / jnp.expand_dims(mass, -1)


def init_N_component(state, perm, func, p):
    n = state.mesh.n + tuple([1] * (3 - state.mesh.dim))
    dx = np.array(state.mesh.dx)
    dx /= dx.min()  # rescale dx to avoid NaNs when using single precision

    shape = [i * 2 if i > 1 else i for i in n]
    ij = [jfft.fftfreq(n, 1 / n, dtype=jnp.float64) for n in shape]
    ij = jnp.meshgrid(*ij, indexing="ij")
    x, y, z = [ij[ind] * dx[ind] for ind in perm]
    Lx = [n[ind] * dx[ind] for ind in perm]
    dx = [dx[ind] for ind in perm]

    # TODO enable pseudo PBCs
    # offsets = [state.arange(-state.mesh.pbc[ind], state.mesh.pbc[ind]+1) for ind in perm] # offset of pseudo PBC images
    # offsets = torch.stack(torch.meshgrid(*offsets, indexing="ij"), dim=-1).flatten(end_dim=-2)
    # Nc = state.zeros(shape)
    # for offset in offsets:
    #    Nc += func(x + offset[0]*Lx[0], y + offset[1]*Lx[1], z + offset[2]*Lx[2], *dx, *dx, self._p)
    Nc = func(x, y, z, *dx, *dx, p)

    dim = [i for i in range(3) if n[i] > 1]
    if len(dim) > 0:
        Nc = jnp.fft.rfftn(Nc, axes=dim)
    return Nc.real.clone()


def init_N(state, p):
    logging.info_green(f"[DemagField]: Set up demag tensor")

    with jax.experimental.enable_x64():
        Nxx = init_N_component(state, [0, 1, 2], demag_f, p).astype(state.dtype)
        Nxy = init_N_component(state, [0, 1, 2], demag_g, p).astype(state.dtype)
        Nxz = init_N_component(state, [0, 2, 1], demag_g, p).astype(state.dtype)
        Nyy = init_N_component(state, [1, 2, 0], demag_f, p).astype(state.dtype)
        Nyz = init_N_component(state, [1, 2, 0], demag_g, p).astype(state.dtype)
        Nzz = init_N_component(state, [2, 0, 1], demag_f, p).astype(state.dtype)

    state.N_demag = [[Nxx, Nxy, Nxz], [Nxy, Nyy, Nyz], [Nxz, Nyz, Nzz]]
