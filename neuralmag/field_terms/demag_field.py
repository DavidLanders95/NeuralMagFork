# SPDX-License-Identifier: MIT

import os
from time import time

from scipy import constants

from neuralmag.common import VectorFunction, config
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
                VectorFunction(state, tensor=config.backend.demag_field.h2d),
            )
        elif state.mesh.dim == 3:
            setattr(
                state,
                self.attr_name("h", name),
                VectorFunction(state, tensor=config.backend.demag_field.h3d),
            )
        else:
            raise
        # fix reference to h_demag in E_demag if suffix is changed
        if name is not None:
            func = state.remap(self.E, {"h_demag": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), func)
        config.backend.demag_field.init_N(state, self._p)

    @staticmethod
    def e_expr(m, dim):
        rho = Variable("rho", "c" * dim)
        Ms = Variable("material__Ms", "c" * dim)
        h_demag = Variable("h_demag", "n" * dim, (3,))
        return -0.5 * constants.mu_0 * Ms * m.dot(h_demag) * dV()
