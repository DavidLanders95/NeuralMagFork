from ..common import logging, Function, CellFunction, VectorFunction
from .field_term import FieldTerm
import numpy as np
from scipy import constants
import torch
import torch.fft
from torch import asinh, atan, sqrt, log, abs
from time import time
import os

__all__ = ["DemagField"]

def newell_f(points):
    x = abs(points[:,:,:,0])
    y = abs(points[:,:,:,1])
    z = abs(points[:,:,:,2])

    result = 1.0 / 6.0 * (2*x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)

    mask = (x**2 + z**2).gt(0)
    result[mask] += (y / 2.0 * (z**2 - x**2) * asinh(y / sqrt(x**2 + z**2)))[mask]

    mask = (x**2 + y**2).gt(0)
    result[mask] += (z / 2.0 * (y**2 - x**2) * asinh(z / sqrt(x**2 + y**2)))[mask]

    mask = (x * (x**2 + y**2 + z**2)).gt(0)
    result[mask] -= (x * y * z * atan(y*z / (x * sqrt(x**2 + y**2 + z**2))))[mask]

    return result

def newell_g(points):
    x = points[:,:,:,0]
    y = points[:,:,:,1]
    z = abs(points[:,:,:,2])

    result = - x*y * sqrt(x**2 + y**2 + z**2) / 3.0

    mask = (x**2 + y**2).gt(0) # x**2 + y**2 > 0
    result[mask] += (x*y*z * asinh(z / sqrt(x**2 + y**2)))[mask]

    mask = (y**2 + z**2).gt(0)
    result[mask] += (y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / sqrt(y**2 + z**2)))[mask]

    mask = (x**2 + z**2).gt(0)
    result[mask] += (x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / sqrt(x**2 + z**2)))[mask]

    mask = (z * (x**2 + y**2 + z**2)).ne(0)
    result[mask] -= ( z**3 / 6.0 * atan(x*y / (z * sqrt(x**2 + y**2 + z**2))))[mask]

    mask = (y * (x**2 + y**2 + z**2)).ne(0)
    result[mask] -= (z * y**2 / 2.0 * atan(x*z / (y * sqrt(x**2 + y**2 + z**2))))[mask]

    mask = (x * (x**2 + y**2 + z**2)).ne(0)
    result[mask] -= (z * x**2 / 2.0 * atan(y*z / (x * sqrt(x**2 + y**2 + z**2))))[mask]

    return result


def dipole_f(points):
    x = points[:,:,:,0]
    y = points[:,:,:,1]
    z = points[:,:,:,2]

    result = (2.*x**2 - y**2 - z**2) * pow(x**2 + y**2 + z**2, -5./2.)
    result[0,0,0] = 0.
    return result

def dipole_g(points):
    x = points[:,:,:,0]
    y = points[:,:,:,1]
    z = points[:,:,:,2]

    result = 3.*x*y * pow(x**2 + y**2 + z**2, -5./2.)
    result[0,0,0] = 0.
    return result

class DemagField(FieldTerm):
    def __init__(self, p = 20):
        self._p = p

    def _init_N_component(self, state, perm, func_near, func_far):
        # dipole far-field
        shape = [1 if n==1 else 2*n for n in state.mesh.n]
        ij = [torch.fft.fftshift(torch.arange(n, device=state.device, dtype=state.dtype)) - n//2 for n in shape]
        ij = torch.meshgrid(*ij,indexing='ij')

        r = torch.stack([ij[ind]*state.mesh.dx[ind] for ind in perm], dim=-1)
        Nc = func_far(r) * np.prod(state.mesh.dx) / (4.*np.pi)

        # newell near-field
        n_near = np.minimum(state.mesh.n, self._p)
        N_near = torch.zeros([1 if n==1 else 2*n for n in n_near], device=state.device, dtype=state.dtype)
        ij = [torch.fft.fftshift(torch.arange(n, device=state.device, dtype=state.dtype)) - n//2 for n in N_near.shape[:3]]
        ij = torch.meshgrid(*ij,indexing='ij')

        for kl in np.rollaxis(np.indices((2,)*6), 0, 7).reshape(64, 6):
            k, l = kl[:3], kl[3:]
            r = torch.stack([(ij[ind] + k[ind] - l[ind])*state.mesh.dx[ind] for ind in perm], dim=-1)
            N_near[:,:,:] -= (-1)**np.sum(kl) * func_near(r) / (4.*np.pi*np.prod(state.mesh.dx))

        Nc[:n_near[0]   ,:n_near[1]   ,:n_near[2]   ] = N_near[:n_near[0]   ,:n_near[1]   ,:n_near[2]   ]
        Nc[:n_near[0]   ,:n_near[1]   ,-n_near[2]+1:] = N_near[:n_near[0]   ,:n_near[1]   ,-n_near[2]+1:]
        Nc[:n_near[0]   ,-n_near[1]+1:,:n_near[2]   ] = N_near[:n_near[0]   ,-n_near[1]+1:,:n_near[2]   ]
        Nc[:n_near[0]   ,-n_near[1]+1:,-n_near[2]+1:] = N_near[:n_near[0]   ,-n_near[1]+1:,-n_near[2]+1:]
        Nc[-n_near[0]+1:,:n_near[1]   ,:n_near[2]   ] = N_near[-n_near[0]+1:,:n_near[1]   ,:n_near[2]   ]
        Nc[-n_near[0]+1:,:n_near[1]   ,-n_near[2]+1:] = N_near[-n_near[0]+1:,:n_near[1]   ,-n_near[2]+1:]
        Nc[-n_near[0]+1:,-n_near[1]+1:,:n_near[2]   ] = N_near[-n_near[0]+1:,-n_near[1]+1:,:n_near[2]   ]
        Nc[-n_near[0]+1:,-n_near[1]+1:,-n_near[2]+1:] = N_near[-n_near[0]+1:,-n_near[1]+1:,-n_near[2]+1:]

        return torch.fft.rfftn(Nc, dim = [i for i in range(3) if state.mesh.n[i] > 1]).real.clone()

    def _init_N(self, state):
        time_kernel = time()
        Nxx = self._init_N_component(state, [0,1,2], newell_f, dipole_f)
        Nxy = self._init_N_component(state, [0,1,2], newell_g, dipole_g)
        Nxz = self._init_N_component(state, [0,2,1], newell_g, dipole_g)
        Nyy = self._init_N_component(state, [1,2,0], newell_f, dipole_f)
        Nyz = self._init_N_component(state, [1,2,0], newell_g, dipole_g)
        Nzz = self._init_N_component(state, [2,0,1], newell_f, dipole_f)

        self._N = [[Nxx, Nxy, Nxz],
                   [Nxy, Nyy, Nyz],
                   [Nxz, Nyz, Nzz]]
        logging.info_green(f"[DEMAG]: Time calculation of demag kernel = {time() - time_kernel} s")

    def _hcell(self, state, m):
        if not hasattr(self, "_N"):
            self._init_N(state)

        hx = torch.zeros(list(self._N[0][0].shape), device=state.device, dtype=torch.complex128)
        hy = torch.zeros(list(self._N[0][0].shape), device=state.device, dtype=torch.complex128)
        hz = torch.zeros(list(self._N[0][0].shape), device=state.device, dtype=torch.complex128)
        for ax in range(3):
            m_pad_fft1D = torch.fft.rfftn(state.material.Ms.tensor.unsqueeze(-1) * m[:,:,:,(ax,)], dim = [i for i in range(3) if state.mesh.n[i] > 1], s = [2*state.mesh.n[i] for i in range(3) if state.mesh.n[i] > 1]).squeeze(-1)

            hx += self._N[0][ax] * m_pad_fft1D
            hy += self._N[1][ax] * m_pad_fft1D
            hz += self._N[2][ax] * m_pad_fft1D

        hx = torch.fft.irfftn(hx, dim = [i for i in range(3) if state.mesh.n[i] > 1])
        hy = torch.fft.irfftn(hy, dim = [i for i in range(3) if state.mesh.n[i] > 1])
        hz = torch.fft.irfftn(hz, dim = [i for i in range(3) if state.mesh.n[i] > 1])

        return torch.stack([hx[:state.mesh.n[0],:state.mesh.n[1],:state.mesh.n[2]],
                            hy[:state.mesh.n[0],:state.mesh.n[1],:state.mesh.n[2]],
                            hz[:state.mesh.n[0],:state.mesh.n[1],:state.mesh.n[2]]], dim=3)

    def h(self, state):
        mcell = state.m.tensor[1: , 1: , 1: , :] \
              + state.m.tensor[:-1, 1: , 1: , :] \
              + state.m.tensor[1: , :-1, 1: , :] \
              + state.m.tensor[:-1, :-1, 1: , :] \
              + state.m.tensor[1: , 1: , :-1, :] \
              + state.m.tensor[:-1, 1: , :-1, :] \
              + state.m.tensor[1: , :-1, :-1, :] \
              + state.m.tensor[:-1, :-1, :-1, :]
        mcell /= 8.

        hcell = self._hcell(state, mcell)

        h = VectorFunction(state)
        h.tensor[:-1,:-1,:-1] += 1./8. * hcell
        h.tensor[:-1,:-1, 1:] += 1./8. * hcell
        h.tensor[:-1, 1:,:-1] += 1./8. * hcell
        h.tensor[:-1, 1:, 1:] += 1./8. * hcell
        h.tensor[ 1:,:-1,:-1] += 1./8. * hcell
        h.tensor[ 1:,:-1, 1:] += 1./8. * hcell
        h.tensor[ 1:, 1:,:-1] += 1./8. * hcell
        h.tensor[ 1:, 1:, 1:] += 1./8. * hcell

        ones = CellFunction(state).from_constant(1.).tensor
        mass = Function(state).tensor

        mass[:-1,:-1,:-1] += 1./8. * ones
        mass[:-1,:-1, 1:] += 1./8. * ones
        mass[:-1, 1:,:-1] += 1./8. * ones
        mass[:-1, 1:, 1:] += 1./8. * ones

        mass[ 1:,:-1,:-1] += 1./8. * ones
        mass[ 1:,:-1, 1:] += 1./8. * ones
        mass[ 1:, 1:,:-1] += 1./8. * ones
        mass[ 1:, 1:, 1:] += 1./8. * ones

        h.tensor.multiply_(1. / mass.unsqueeze(-1))
        return h
