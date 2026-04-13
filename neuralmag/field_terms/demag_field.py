# SPDX-License-Identifier: MIT


from scipy import constants

from neuralmag.common import VectorCellFunction, VectorFunction, config
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

    def __init__(self, p=20, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self._p = p
        self._batch_size = batch_size

    def register(self, state, name=None):
        super().register(state, name)
        m_spaces = state.m.spaces
        pbc = state.mesh.pbc
        is_true_pbc = all(pbc[i] == float("inf") for i in range(state.mesh.dim))

        if is_true_pbc:
            if state.mesh.dim < 3:
                raise ValueError(
                    f"True PBC demag field requires a 3D mesh, got {state.mesh.dim}D."
                )
            if set(m_spaces) == {"c"}:
                setattr(
                    state,
                    self.attr_name("h", name),
                    VectorCellFunction(
                        state,
                        tensor=config.backend.demag_field.h_cell_pbc,
                    ),
                )
            elif set(m_spaces) == {"n"}:
                setattr(
                    state,
                    self.attr_name("h", name),
                    VectorFunction(
                        state,
                        tensor=config.backend.demag_field.h3d_pbc,
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported discretization '{m_spaces}' for true PBC demag."
                )
        elif set(m_spaces) == {"c"}:
            dim = state.mesh.dim
            h_cell = config.backend.demag_field.h_cell
            if dim < 3:
                pad = (None,) * (3 - dim)

                def h_func(N_demag, m, material__Ms, rho):
                    m = m[(...,) + pad + (slice(None),)]
                    rho = rho[(...,) + pad]
                    Ms = material__Ms[(...,) + pad]
                    h = h_cell(N_demag, m, Ms, rho)
                    return h[(slice(None),) * dim + (0,) * (3 - dim) + (slice(None),)]
            else:
                h_func = h_cell
            setattr(
                state,
                self.attr_name("h", name),
                VectorCellFunction(state, tensor=h_func),
            )
        elif set(m_spaces) == {"n"}:
            if any(pbc[i] > 0 for i in range(state.mesh.dim)):
                raise ValueError(
                    "Pseudo PBC demag is only supported for cell-centred discretization."
                )
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
        else:
            raise
        # fix reference to h_demag in E_demag if suffix is changed
        if name is not None:
            func = state.remap(self.E, {"h_demag": self.attr_name("h", name)})
            setattr(state, self.attr_name("E", name), func)
        if not is_true_pbc:
            config.backend.demag_field.init_N(state, self._p, self._batch_size)

    @staticmethod
    def e_expr(m, dim, _options):
        m_spaces = _options["m_spaces"]
        rho = Variable("rho", "c" * dim)
        Ms = Variable("material__Ms", "c" * dim)
        h_demag = Variable("h_demag", m_spaces, (3,))
        return -0.5 * constants.mu_0 * Ms * m.dot(h_demag) * dV()
