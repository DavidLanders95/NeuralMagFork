# MIT License
#
# Copyright (c) 2022-2025 NeuralMag team
#
# This file is part of NeuralMag – a simulation package for inverse micromagnetics.
# Repository: https://gitlab.com/neuralmag/neuralmag
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from time import time

from scipy import constants

from neuralmag.common import config
from neuralmag.common.engine import Variable, dV
from neuralmag.field_terms.field_term import FieldTerm

__all__ = ["DemagField"]


class DemagField(FieldTerm):
    r"""
    Effective field contribution corresponding to the demagnetization field (also referred to as magnetostatic field or stray field).
    The demagnetization field is computed from the scalar potential :math:`u` as

    .. math::

        \vec{H}_\text{demag} = - \nabla u

    with :math:`u` being calculated by the Poisson equation

    .. math::

      \Delta u = \nabla \cdot (M_s \vec{m})

    with open boundary conditions.


    :param p: Distance threshhold at which the demag tensor is approximated
              by a dipole field given in numbers of cells. Defaults to 20.
    :type p: int
    :param n_gauss: Degree of Gauss quadrature used in the form compiler.
    :type n_gauss: int

    :Required state attributes (if not renamed):
        * **state.material.Ms** (*cell scalar field*) The saturation magnetization in A/m
    """
    default_name = "demag"
    h = None

    def __init__(self, p=20, **kwargs):
        super().__init__(**kwargs)
        self._p = p

    def register(self, state, name=None):
        super().register(state, name)
        if state.mesh.dim == 2:
            setattr(
                state,
                self.attr_name("h", name),
                (config.backend.demag_field.h2d, "nn", (3,)),
            )
        elif state.mesh.dim == 3:
            setattr(
                state,
                self.attr_name("h", name),
                (config.backend.demag_field.h3d, "nnn", (3,)),
            )
        else:
            raise
        # fix reference to h_demag in E_demag if suffix is changed
        if name is not None:
            wrapped = state.wrap_func(self.E, {"h_demag": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), wrapped)
        config.backend.demag_field.init_N(state, self._p)

    @staticmethod
    def e_expr(m, dim):
        rho = Variable("rho", "c" * dim)
        Ms = Variable("material__Ms", "c" * dim)
        h_demag = Variable("h_demag", "n" * dim, (3,))
        return -0.5 * constants.mu_0 * Ms * m.dot(h_demag) * dV()
