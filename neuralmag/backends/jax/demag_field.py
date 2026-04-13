# SPDX-License-Identifier: MIT


import numpy as np

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from jax.numpy import abs, pi, sqrt
from jax.numpy import arcsinh as asinh
from jax.numpy import arctan as atan
from neuralmag.common import logging

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


def h_cell(N_demag, m, material__Ms, rho):
    dim = [i for i in range(3) if m.shape[i] > 1]

    if len(dim) == 0:
        hx = jnp.zeros_like(m[:, :, :, 0])
        hy = jnp.zeros_like(m[:, :, :, 0])
        hz = jnp.zeros_like(m[:, :, :, 0])
        for ax in range(3):
            mx = rho * material__Ms * m[:, :, :, ax]
            hx += N_demag[0][ax] * mx
            hy += N_demag[1][ax] * mx
            hz += N_demag[2][ax] * mx
        return jnp.stack([hx, hy, hz], axis=3)

    N_shape = N_demag[0][0].shape
    s = [
        N_shape[i]
        if i != dim[-1]
        else (2 * m.shape[i] if N_shape[i] == m.shape[i] + 1 else m.shape[i])
        for i in dim
    ]

    hx = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[m.dtype])
    hy = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[m.dtype])
    hz = jnp.zeros(N_demag[0][0].shape, dtype=complex_dtype[m.dtype])

    for ax in range(3):
        m_pad_fft1D = jnp.fft.rfftn(rho * material__Ms * m[:, :, :, ax], axes=dim, s=s)
        hx += N_demag[0][ax] * m_pad_fft1D
        hy += N_demag[1][ax] * m_pad_fft1D
        hz += N_demag[2][ax] * m_pad_fft1D

    hx = jnp.fft.irfftn(hx, axes=dim, s=s)
    hy = jnp.fft.irfftn(hy, axes=dim, s=s)
    hz = jnp.fft.irfftn(hz, axes=dim, s=s)

    return jnp.stack(
        [
            hx[: m.shape[0], : m.shape[1], : m.shape[2]],
            hy[: m.shape[0], : m.shape[1], : m.shape[2]],
            hz[: m.shape[0], : m.shape[1], : m.shape[2]],
        ],
        axis=3,
    )


def h_cell_pbc(m, material__Ms, rho, dx):
    """True PBC demag field via k-space Poisson solver (cell-centred)."""
    n = m.shape[:3]
    dim = [i for i in range(3) if n[i] > 1]
    Ms = jnp.expand_dims(rho * material__Ms, -1)

    kx = (2.0 * pi * jnp.arange(n[0]) / n[0]).reshape(-1, 1, 1)
    ky = (2.0 * pi * jnp.arange(n[1]) / n[1]).reshape(1, -1, 1)
    kz = (2.0 * pi * jnp.arange(n[2]) / n[2]).reshape(1, 1, -1)

    M_fft = jnp.fft.fftn(Ms * m, axes=dim)

    div_M = (
        (1.0 - jnp.exp(-1j * kx)) * M_fft[..., 0] / dx[0]
        + (1.0 - jnp.exp(-1j * ky)) * M_fft[..., 1] / dx[1]
        + (1.0 - jnp.exp(-1j * kz)) * M_fft[..., 2] / dx[2]
    )

    laplacian = (
        4.0 / dx[0] ** 2 * jnp.sin(kx / 2.0) ** 2
        + 4.0 / dx[1] ** 2 * jnp.sin(ky / 2.0) ** 2
        + 4.0 / dx[2] ** 2 * jnp.sin(kz / 2.0) ** 2
    )
    laplacian = laplacian.at[0, 0, 0].set(1.0)

    u_fft = -div_M / laplacian
    u_fft = u_fft.at[0, 0, 0].set(0.0)

    h_fft = jnp.stack(
        [
            (1.0 - jnp.exp(1j * kx)) * u_fft / dx[0],
            (1.0 - jnp.exp(1j * ky)) * u_fft / dx[1],
            (1.0 - jnp.exp(1j * kz)) * u_fft / dx[2],
        ],
        axis=3,
    )

    return jnp.fft.ifftn(h_fft, axes=dim).real


def init_N_component(state, perm, func, p, batch_size=1):
    n = state.mesh.n + tuple([1] * (3 - state.mesh.dim))
    pbc = state.mesh.pbc + tuple([0] * (3 - state.mesh.dim))
    dx = np.array(state.mesh.dx)
    dx /= dx.min()  # rescale dx to avoid NaNs when using single precision

    shape = tuple(
        1 if n[i] == 1 else (n[i] if pbc[i] > 0 else 2 * n[i]) for i in range(3)
    )
    ij = [jfft.fftfreq(s, 1 / s, dtype=jnp.float64) for s in shape]
    ij = jnp.meshgrid(*ij, indexing="ij")
    x, y, z = [ij[ind] * dx[ind] for ind in perm]
    Lx = [n[ind] * dx[ind] for ind in perm]
    dx = [dx[ind] for ind in perm]

    offsets = [np.arange(-pbc[ind], pbc[ind] + 1) for ind in perm]
    offsets = np.stack(np.meshgrid(*offsets, indexing="ij"), axis=-1).reshape(-1, 3)

    Nc = jnp.zeros(shape, dtype=jnp.float64)
    for i in range(0, len(offsets), batch_size):
        chunk = jnp.array(offsets[i : i + batch_size]).reshape(-1, 1, 1, 1, 3)
        xx = x[None] + chunk[..., 0] * Lx[0]
        yy = y[None] + chunk[..., 1] * Lx[1]
        zz = z[None] + chunk[..., 2] * Lx[2]
        Nc += func(xx, yy, zz, *dx, *dx, p).sum(0)

    dim = [i for i in range(3) if n[i] > 1]
    if len(dim) > 0:
        Nc = jnp.fft.rfftn(Nc, axes=dim)
    return Nc.real.copy()


def init_N(state, p, batch_size=1):
    logging.info_green("[DemagField]: Set up demag tensor")

    with jax.enable_x64():
        Nxx = init_N_component(state, [0, 1, 2], demag_f, p, batch_size).astype(
            state.dtype
        )
        Nxy = init_N_component(state, [0, 1, 2], demag_g, p, batch_size).astype(
            state.dtype
        )
        Nxz = init_N_component(state, [0, 2, 1], demag_g, p, batch_size).astype(
            state.dtype
        )
        Nyy = init_N_component(state, [1, 2, 0], demag_f, p, batch_size).astype(
            state.dtype
        )
        Nyz = init_N_component(state, [1, 2, 0], demag_g, p, batch_size).astype(
            state.dtype
        )
        Nzz = init_N_component(state, [2, 0, 1], demag_f, p, batch_size).astype(
            state.dtype
        )

    state.N_demag = [[Nxx, Nxy, Nxz], [Nxy, Nyy, Nyz], [Nxz, Nyz, Nzz]]
