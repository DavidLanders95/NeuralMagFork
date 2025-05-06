# SPDX-License-Identifier: MIT

import numpy as np
import torch
import torch.fft
from scipy import constants
from torch import abs, asinh, atan, log, sqrt

from neuralmag.common import logging

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
    result += (y / 2.0 * (z2 - x2) * asinh(y / sqrt(x2 + z2))).nan_to_num(
        posinf=0, neginf=0
    )
    result += (z / 2.0 * (y2 - x2) * asinh(z / sqrt(x2 + y2))).nan_to_num(
        posinf=0, neginf=0
    )
    result -= (x * y * z * atan(y * z / (x * r))).nan_to_num(posinf=0, neginf=0)
    return result


def g(x, y, z):
    z = abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    r = sqrt(x2 + y2 + z2)
    result = -x * y * r / 3.0
    result += (x * y * z * asinh(z / sqrt(x2 + y2))).nan_to_num(posinf=0, neginf=0)
    result += (y / 6.0 * (3.0 * z2 - y2) * asinh(x / sqrt(y2 + z2))).nan_to_num(
        posinf=0, neginf=0
    )
    result += (x / 6.0 * (3.0 * z2 - x2) * asinh(y / sqrt(x2 + z2))).nan_to_num(
        posinf=0, neginf=0
    )
    result -= (z**3 / 6.0 * atan(x * y / (z * r))).nan_to_num(posinf=0, neginf=0)
    result -= (z * y2 / 2.0 * atan(x * z / (y * r))).nan_to_num(posinf=0, neginf=0)
    result -= (z * x2 / 2.0 * atan(y * z / (x * r))).nan_to_num(posinf=0, neginf=0)
    return result


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
    ret = (
        F0(func, x, y, z, dy, dY, dz, dZ)
        - F0(func, x - dx, y, z, dy, dY, dz, dZ)
        - F0(func, x + dX, y, z, dy, dY, dz, dZ)
        + F0(func, x - dx + dX, y, z, dy, dY, dz, dZ)
    )
    return -ret / (4.0 * np.pi * dx * dy * dz)


def dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    result = (2.0 * x**2 - y**2 - z**2) * pow(
        x**2 + y**2 + z**2, -5.0 / 2.0
    )
    result[0, 0, 0] = 0.0
    return result * dx * dy * dz / (4.0 * np.pi)


def dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ):
    z = z + dZ / 2.0 - dz / 2.0  # diff of cell centers for non-equidistant demag
    result = 3.0 * x * y * pow(x**2 + y**2 + z**2, -5.0 / 2.0)
    result[0, 0, 0] = 0.0
    return result * dx * dy * dz / (4.0 * np.pi)


def demag_f(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_f(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(
        dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2
    ) < p**2
    res[near] = newell(f, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ)
    return res


def demag_g(x, y, z, dx, dy, dz, dX, dY, dZ, p):
    res = dipole_g(x, y, z, dx, dy, dz, dX, dY, dZ)
    near = (x**2 + y**2 + z**2) / max(
        dx**2 + dy**2 + dz**2, dX**2 + dY**2 + dZ**2
    ) < p**2
    res[near] = newell(g, x[near], y[near], z[near], dx, dy, dz, dX, dY, dZ)
    return res


def h_cell(N_demag, mcell, material__Ms, rho):
    dim = [i for i in range(3) if mcell.shape[i] > 1]
    s = [mcell.shape[i] * 2 for i in dim]

    hx = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[mcell.dtype])
    hy = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[mcell.dtype])
    hz = torch.zeros_like(N_demag[0][0], dtype=complex_dtype[mcell.dtype])
    for ax in range(3):
        m_pad_fft1D = torch.fft.rfftn(
            rho * material__Ms * mcell[:, :, :, ax], dim=dim, s=s
        )
        hx += N_demag[0][ax] * m_pad_fft1D
        hy += N_demag[1][ax] * m_pad_fft1D
        hz += N_demag[2][ax] * m_pad_fft1D

    hx = torch.fft.irfftn(hx, dim=dim)
    hy = torch.fft.irfftn(hy, dim=dim)
    hz = torch.fft.irfftn(hz, dim=dim)

    return torch.stack(
        [
            hx[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
            hy[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
            hz[: mcell.shape[0], : mcell.shape[1], : mcell.shape[2]],
        ],
        dim=3,
    )


def h2d(N_demag, m, material__Ms, rho):
    mcell = (m[1:, 1:, :] + m[:-1, 1:, :] + m[1:, :-1, :] + m[:-1, :-1, :]).unsqueeze(
        -2
    ) / 4.0

    # TODO behavior for unsqueezed scalars seems OK, but not so elegant?
    Ms_hcell = (rho * material__Ms).unsqueeze(-1) * h_cell(
        N_demag, mcell, material__Ms.unsqueeze(-1), rho.unsqueeze(-1)
    ).squeeze(-2)

    h = torch.zeros(m.shape, dtype=m.dtype, device=m.device)
    h[:-1, :-1] += Ms_hcell
    h[:-1, 1:] += Ms_hcell
    h[1:, :-1] += Ms_hcell
    h[1:, 1:] += Ms_hcell

    mass = torch.zeros(h.shape[:-1], dtype=h.dtype, device=h.device)
    mass[:-1, :-1] += rho * material__Ms
    mass[:-1, 1:] += rho * material__Ms
    mass[1:, :-1] += rho * material__Ms
    mass[1:, 1:] += rho * material__Ms

    return h / mass.unsqueeze(-1)


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

    Ms_hcell = (rho * material__Ms).unsqueeze(-1) * h_cell(
        N_demag, mcell, material__Ms, rho
    )

    h = torch.zeros(m.shape, dtype=m.dtype, device=m.device)
    h[:-1, :-1, :-1] += Ms_hcell
    h[:-1, :-1, 1:] += Ms_hcell
    h[:-1, 1:, :-1] += Ms_hcell
    h[:-1, 1:, 1:] += Ms_hcell
    h[1:, :-1, :-1] += Ms_hcell
    h[1:, :-1, 1:] += Ms_hcell
    h[1:, 1:, :-1] += Ms_hcell
    h[1:, 1:, 1:] += Ms_hcell

    mass = torch.zeros(h.shape[:-1], dtype=h.dtype, device=h.device)
    mass[:-1, :-1, :-1] += rho * material__Ms
    mass[:-1, :-1, 1:] += rho * material__Ms
    mass[:-1, 1:, :-1] += rho * material__Ms
    mass[:-1, 1:, 1:] += rho * material__Ms
    mass[1:, :-1, :-1] += rho * material__Ms
    mass[1:, :-1, 1:] += rho * material__Ms
    mass[1:, 1:, :-1] += rho * material__Ms
    mass[1:, 1:, 1:] += rho * material__Ms

    return h / mass.unsqueeze(-1)


def init_N_component(state, perm, func, p):
    n = state.mesh.n + tuple([1] * (3 - state.mesh.dim))
    dx = np.array(state.mesh.dx)
    dx /= dx.min()  # rescale dx to avoid NaNs when using single precision

    shape = [i * 2 if i > 1 else i for i in n]
    ij = [
        torch.fft.fftfreq(n, 1 / n).to(dtype=torch.float64, device=state.device)
        for n in shape
    ]  # local indices
    ij = torch.meshgrid(*ij, indexing="ij")
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
        Nc = torch.fft.rfftn(Nc, dim=dim)
    return Nc.real.clone()


def init_N(state, p):
    logging.info_green(f"[DemagField]: Set up demag tensor")

    Nxx = init_N_component(state, [0, 1, 2], demag_f, p).to(dtype=state.dtype)
    Nxy = init_N_component(state, [0, 1, 2], demag_g, p).to(dtype=state.dtype)
    Nxz = init_N_component(state, [0, 2, 1], demag_g, p).to(dtype=state.dtype)
    Nyy = init_N_component(state, [1, 2, 0], demag_f, p).to(dtype=state.dtype)
    Nyz = init_N_component(state, [1, 2, 0], demag_g, p).to(dtype=state.dtype)
    Nzz = init_N_component(state, [2, 0, 1], demag_f, p).to(dtype=state.dtype)

    state.N_demag = [[Nxx, Nxy, Nxz], [Nxy, Nyy, Nyz], [Nxz, Nyz, Nzz]]
