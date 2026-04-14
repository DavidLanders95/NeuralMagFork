# SPDX-License-Identifier: MIT

import numpy as np

import torch
import torch.fft
from neuralmag.common import logging
from torch import abs, asinh, atan, sqrt

complex_dtype = {
    torch.float: torch.complex,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def f(x, y, z):
    x, y, z = abs(x), abs(y), abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = sqrt(x2 + y2 + z2)
    result = 1.0 / 6.0 * (2 * x2 - y2 - z2) * r
    result += (y / 2.0 * (z2 - x2) * asinh(y / sqrt(x2 + z2))).nan_to_num(posinf=0, neginf=0)
    result += (z / 2.0 * (y2 - x2) * asinh(z / sqrt(x2 + y2))).nan_to_num(posinf=0, neginf=0)
    result -= (x * y * z * atan(y * z / (x * r))).nan_to_num(posinf=0, neginf=0)
    return result


def g(x, y, z):
    z = abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = sqrt(x2 + y2 + z2)
    result = -x * y * r / 3.0
    result += (x * y * z * asinh(z / sqrt(x2 + y2))).nan_to_num(posinf=0, neginf=0)
    result += (y / 6.0 * (3.0 * z2 - y2) * asinh(x / sqrt(y2 + z2))).nan_to_num(posinf=0, neginf=0)
    result += (x / 6.0 * (3.0 * z2 - x2) * asinh(y / sqrt(x2 + z2))).nan_to_num(posinf=0, neginf=0)
    result -= (z**3 / 6.0 * atan(x * y / (z * r))).nan_to_num(posinf=0, neginf=0)
    result -= (z * y2 / 2.0 * atan(x * z / (y * r))).nan_to_num(posinf=0, neginf=0)
    result -= (z * x2 / 2.0 * atan(y * z / (x * r))).nan_to_num(posinf=0, neginf=0)
    return result


def F1(func, x, y, z, dz, dZ):
    return func(x, y, z + dZ) - func(x, y, z) - func(x, y, z - dz + dZ) + func(x, y, z - dz)


def F0(func, x, y, z, dy, dY, dz, dZ):
    return (
        F1(func, x, y + dY, z, dz, dZ)
        - F1(func, x, y, z, dz, dZ)
        - F1(func, x, y - dy + dY, z, dz, dZ)
        + F1(func, x, y - dy, z, dz, dZ)
    )


def newell(func, x, y, z, dx, dy, dz, dX, dY, dZ):
    ret = (
        F0(func, x, y, z, dy, dY, dz, dZ)
        - F0(func, x - dx, y, z, dy, dY, dz, dZ)
        - F0(func, x + dX, y, z, dy, dY, dz, dZ)
        + F0(func, x - dx + dX, y, z, dy, dY, dz, dZ)
    )
    return -ret / (4.0 * np.pi * dx * dy * dz)


def dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    result = (2.0 * x**2 - y**2 - z**2) * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    result[0, 0, 0] = 0.0
    return result * dx * dy * dz / (4.0 * np.pi)


def dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    result = 3.0 * x * y * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    result[0, 0, 0] = 0.0
    return result * dx * dy * dz / (4.0 * np.pi)


def demag_f(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2) < p**2
    res[near] = newell(f, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ)
    return res


def demag_g(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2) < p**2
    res[near] = newell(g, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ)
    return res


def h_cell(N_demag, m, material__Ms, rho):
    dim = [i for i in range(3) if m.shape[i] > 1]

    if len(dim) == 0:  # single spin (torch rfftn rejects empty axes, pytorch#96518)
        return torch.stack([N_demag[i][i] * rho * material__Ms * m[..., i] for i in range(3)], dim=-1)

    N_shape = N_demag[0][0].shape
    # Derive FFT size from N_demag shape. For the last (rfftn) axis the
    # stored size is n//2+1 for open BC (2n padded) or n for PBC.
    s = [N_shape[i] if i != dim[-1] else (2 * m.shape[i] if N_shape[i] == m.shape[i] + 1 else m.shape[i]) for i in dim]

    hx = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[m.dtype])
    hy = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[m.dtype])
    hz = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[m.dtype])
    for ax in range(3):
        m_pad_fft1D = torch.fft.rfftn(rho * material__Ms * m[:, :, :, ax], dim=dim, s=s)
        hx += N_demag[0][ax] * m_pad_fft1D
        hy += N_demag[1][ax] * m_pad_fft1D
        hz += N_demag[2][ax] * m_pad_fft1D

    hx = torch.fft.irfftn(hx, dim=dim, s=s)
    hy = torch.fft.irfftn(hy, dim=dim, s=s)
    hz = torch.fft.irfftn(hz, dim=dim, s=s)

    return torch.stack(
        [
            hx[: m.shape[0], : m.shape[1], : m.shape[2]],
            hy[: m.shape[0], : m.shape[1], : m.shape[2]],
            hz[: m.shape[0], : m.shape[1], : m.shape[2]],
        ],
        dim=3,
    )


def h_cell_pbc(m, material__Ms, rho, dx):
    """True PBC demag field via k-space Poisson solver (cell-centred)."""
    n = m.shape[:3]
    dim = [i for i in range(3) if n[i] > 1]
    Ms = (rho * material__Ms).unsqueeze(-1)

    kx = (2.0 * np.pi * torch.arange(n[0], device=m.device, dtype=m.dtype) / n[0]).reshape(-1, 1, 1)
    ky = (2.0 * np.pi * torch.arange(n[1], device=m.device, dtype=m.dtype) / n[1]).reshape(1, -1, 1)
    kz = (2.0 * np.pi * torch.arange(n[2], device=m.device, dtype=m.dtype) / n[2]).reshape(1, 1, -1)

    M_fft = torch.fft.fftn(Ms * m, dim=dim)

    div_M = (
        (1.0 - torch.exp(-1j * kx)) * M_fft[..., 0] / dx[0]
        + (1.0 - torch.exp(-1j * ky)) * M_fft[..., 1] / dx[1]
        + (1.0 - torch.exp(-1j * kz)) * M_fft[..., 2] / dx[2]
    )

    laplacian = (
        4.0 / dx[0] ** 2 * torch.sin(kx / 2.0) ** 2
        + 4.0 / dx[1] ** 2 * torch.sin(ky / 2.0) ** 2
        + 4.0 / dx[2] ** 2 * torch.sin(kz / 2.0) ** 2
    )
    laplacian[0, 0, 0] = 1.0

    u_fft = -div_M / laplacian
    u_fft[0, 0, 0] = 0.0

    h_fft = torch.stack(
        [
            (1.0 - torch.exp(1j * kx)) * u_fft / dx[0],
            (1.0 - torch.exp(1j * ky)) * u_fft / dx[1],
            (1.0 - torch.exp(1j * kz)) * u_fft / dx[2],
        ],
        dim=3,
    )

    return torch.fft.ifftn(h_fft, dim=dim).real


def init_N_component(state, perm, func, p, batch_size=1):
    n = state.mesh.n + tuple([1] * (3 - state.mesh.dim))
    pbc = state.mesh.pbc + tuple([0] * (3 - state.mesh.dim))
    dx = np.array(state.mesh.dx)
    dx /= dx.min()  # rescale dx to avoid NaNs when using single precision

    shape = tuple(1 if n[i] == 1 else (n[i] if pbc[i] > 0 else 2 * n[i]) for i in range(3))
    ij = [torch.fft.fftfreq(s, 1 / s).to(dtype=torch.float64, device=state.device) for s in shape]
    ij = torch.meshgrid(*ij, indexing="ij")
    x, y, z = [ij[ind] * dx[ind] for ind in perm]
    Lx = [n[ind] * dx[ind] for ind in perm]
    dx = [dx[ind] for ind in perm]

    offsets = [np.arange(-pbc[ind], pbc[ind] + 1) for ind in perm]
    offsets = np.stack(np.meshgrid(*offsets, indexing="ij"), axis=-1).reshape(-1, 3)

    Nc = torch.zeros(shape, dtype=torch.float64, device=state.device)
    for i in range(0, len(offsets), batch_size):
        chunk = torch.tensor(
            offsets[i : i + batch_size],
            dtype=torch.float64,
            device=state.device,
        ).reshape(-1, 1, 1, 1, 3)
        xx = x[None] + chunk[..., 0] * Lx[0]
        yy = y[None] + chunk[..., 1] * Lx[1]
        zz = z[None] + chunk[..., 2] * Lx[2]
        Nc += func(xx, yy, zz, *dx, *dx, p).sum(0)

    dim = [i for i in range(3) if n[i] > 1]
    if len(dim) > 0:
        Nc = torch.fft.rfftn(Nc, dim=dim)
    return Nc.real.clone()


def init_N(state, p, batch_size=1):
    logging.info_green("[DemagField]: Set up demag tensor")

    Nxx = init_N_component(state, [0, 1, 2], demag_f, p, batch_size).to(dtype=state.dtype)
    Nxy = init_N_component(state, [0, 1, 2], demag_g, p, batch_size).to(dtype=state.dtype)
    Nxz = init_N_component(state, [0, 2, 1], demag_g, p, batch_size).to(dtype=state.dtype)
    Nyy = init_N_component(state, [1, 2, 0], demag_f, p, batch_size).to(dtype=state.dtype)
    Nyz = init_N_component(state, [1, 2, 0], demag_g, p, batch_size).to(dtype=state.dtype)
    Nzz = init_N_component(state, [2, 0, 1], demag_f, p, batch_size).to(dtype=state.dtype)

    state.N_demag = [[Nxx, Nxy, Nxz], [Nxy, Nyy, Nyz], [Nxz, Nyz, Nzz]]
