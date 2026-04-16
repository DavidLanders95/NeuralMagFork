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

    with open boundary conditions by default. Periodic boundary conditions
    can be enabled on the :class:`~neuralmag.Mesh` via ``pbc``: true PBC
    (requires a 3D mesh with all three directions set to ``True``/``inf``)
    switches the convolution to the periodic kernel of Bruckner et al.,
    *Sci. Rep.* **11**, 9202 (2021); pseudo-PBC reuses the open-boundary
    kernel with image copies. See the :ref:`PBC user guide <pbc>` for
    details.


    :param p: Distance threshhold at which the demag tensor is approximated
              by a dipole field given in numbers of cells. Defaults to 20.
    :type p: int
    :param batch_size: Batch size for pseudo-PBC image offset computation.
    :type batch_size: int
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
        dim = state.mesh.dim
        pbc = state.mesh.pbc
        is_true_pbc = all(pbc[i] == float("inf") for i in range(dim))

        h_cell = config.backend.demag_field.h_cell

        if is_true_pbc:
            if dim < 3:
                raise ValueError(f"True PBC demag field requires a 3D mesh, got {dim}D.")
            h_cell_kernel = config.backend.demag_field.h_cell_pbc
        elif dim < 3:
            pad = (None,) * (3 - dim)
            crop = (slice(None),) * dim + (0,) * (3 - dim)

            def h_cell_kernel(N_demag, m, material__Ms, rho):
                return h_cell(
                    N_demag,
                    m[(...,) + pad + (slice(None),)],
                    material__Ms[(...,) + pad],
                    rho[(...,) + pad],
                )[crop + (slice(None),)]
        else:
            h_cell_kernel = h_cell

        if set(m_spaces) == {"c"}:
            setattr(
                state,
                self.attr_name("h", name),
                VectorCellFunction(state, tensor=h_cell_kernel),
            )
        elif set(m_spaces) == {"n"}:
            _cell_template = VectorCellFunction(state)
            to_cell_fn = state.resolve(state.m._code.to_cell, ["f"])
            to_node_w_fn = state.resolve(_cell_template._code.to_node_w, ["f", "weight"])

            if is_true_pbc:

                def h_func(m, material__Ms, rho, dx):
                    m_cell = to_cell_fn(m)
                    hc = h_cell_kernel(m_cell, material__Ms, rho, dx)
                    return to_node_w_fn(hc, material__Ms)

            else:

                def h_func(N_demag, m, material__Ms, rho):
                    m_cell = to_cell_fn(m)
                    hc = h_cell_kernel(N_demag, m_cell, material__Ms, rho)
                    return to_node_w_fn(hc, material__Ms)

            setattr(
                state,
                self.attr_name("h", name),
                VectorFunction(state, tensor=h_func),
            )
        else:
            raise ValueError(f"Unsupported discretization '{m_spaces}' for demag.")

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
