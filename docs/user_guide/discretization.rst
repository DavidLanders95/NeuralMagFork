.. _nodal_fd:

Discretization
==============

NeuralMag offers two closely related spatial discretizations on a regular
cuboid grid. Both share the same form compiler and the same FFT
demagnetization kernel; they differ only in *where* the magnetization lives:

==================  ==========================  ================================
Scheme              ``m`` lives on              Field ``h`` is computed on
==================  ==========================  ================================
Nodal FEM           nodes (trilinear basis)     nodes
FIC (cell-centred)  cells (piecewise constant)  nodes, projected back to cells
==================  ==========================  ================================

Pick whichever is more natural for the problem at hand — see the
:ref:`choosing_a_scheme` section below — and then forget about the difference:
the rest of the API is identical.

.. _disc_mesh:

The mesh and notation
---------------------

A :class:`~neuralmag.Mesh` is a regular cuboid grid characterised by

* the number of cells in each principal direction, ``n = (n_x, n_y, n_z)``;
* the cell size, ``dx = (dx_x, dx_y, dx_z)``;
* an optional ``origin``.

Throughout this page we use :math:`i, j, k` for cell indices (running from
``0`` to ``n_*-1``) and :math:`I, J, K` for node indices (running from ``0``
to ``n_*``). Material parameters such as :math:`A_\text{ex}` or
:math:`M_\text{s}` are always piecewise-constant on cells.

.. figure:: ../figures/nodal_fd.png
   :name: fig_nodal_fd
   :scale: 40%

   Regular cuboid grid in NeuralMag.
   (a) 1D representation of the discretization for a continuous field (red)
   and piecewise-constant material parameters (blue).
   (b) 2D trilinear basis function :math:`\phi_i(x_1, x_2)`.

.. _disc_nodal:

Nodal FEM scheme (default)
--------------------------

The default scheme is the *nodal finite-difference* method introduced for
NeuralMag by Abert et al. (*npj Comput. Mater.* **11**, 193, 2025). Despite the
name it is, strictly speaking, a finite-element method on a structured grid
with the standard FD savings: because the mesh is regular, every element
matrix is identical and we never have to assemble or store a global matrix.

Function spaces
^^^^^^^^^^^^^^^

* The magnetization :math:`\vec m` lives in a continuous, piecewise trilinear
  Lagrange space attached to the mesh nodes (basis functions
  :math:`\phi_{IJK}`).
* Material parameters such as :math:`M_\text{s}`, :math:`A_\text{ex}`,
  :math:`K_\text{u}` live in a piecewise-constant space attached to the cells
  (basis functions :math:`\theta_{ijk}`).

This is exactly the standard mixed setup that lets sharp material interfaces
land on cell boundaries while keeping the magnetization continuous across
them — a small but essential property when you compare numerics against
analytical interface models.

Effective field via mass lumping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The discrete effective field at node :math:`i` is obtained from the energy
functional :math:`E[\vec m]` via a mass-lumped weak form:

.. math::

   H_i \;=\; -\,\Bigl[\,\int_{\Omega_\text{m}} \mu_0 M_\text{s}\,
              \vec\phi_i\cdot\vec 1\,\dx\Bigr]^{-1}\,
              \frac{\delta E}{\delta \vec m}\bigl|_{\vec\phi_i},

where the term in square brackets is the diagonal lumped mass at node
:math:`i` and :math:`\delta E/\delta\vec m\bigl|_{\vec\phi_i}` is the Gateaux
derivative of :math:`E` in the direction of the basis function
:math:`\vec\phi_i`. For a quadratic energy (e.g. exchange), this becomes a
constant linear operator

.. math::

   H_i \;=\; \sum_j A_{ij}\, m_j,
   \qquad
   A_{ij} \;=\;
   - \Bigl[\!\int_{\Omega_\text{m}} \mu_0 M_\text{s}\,\vec\phi_i\cdot\vec 1\,\dx\Bigr]^{-1}\!
     \int_{\Omega_\text{m}} A_\text{ex}\,
     \nabla\vec\phi_i : \nabla\vec\phi_j \dx.

Because the grid is regular, :math:`A_{ij}` has a translation-invariant
stencil structure and is **never assembled**: NeuralMag's form compiler emits
a few vectorized array slices that compute :math:`H_i` directly from the
``m`` tensor (see :doc:`form_compiler`). This is what gives the scheme its
finite-difference price tag.

Demag
^^^^^

For the demagnetization field NeuralMag uses the FFT-accelerated convolution
of `magnum.np <https://gitlab.com/magnum.np/magnum.np>`_, the same well
benchmarked kernel as in classical FD codes. Combining the speed of FFT
demag with the interface accuracy of the nodal FEM scheme is the central
practical reason to prefer NeuralMag over a pure FD code.

.. admonition:: When to prefer the nodal scheme
   :class: tip

   * Smooth magnetization fields and well-resolved domain walls.
   * Standard micromagnetic benchmark problems (Standard Problem 4, etc.).
   * Anything that you would solve with a finite-element micromagnetic code
     today and where you want a familiar mental model.

.. _disc_fic:

FIC — cell-centred variant
--------------------------

FIC stores the magnetization at **cell centres** but computes the effective
field with the *same* nodal FEM machinery as above. The bridge between the
two layouts is a pair of mass-lumped :math:`L^2` projections that the form
compiler generates from very short symbolic expressions.

Conceptually, every effective-field evaluation runs three steps:

1. **Cell → node projection** of ``m``: solve

   .. math::

      \int_{\Omega_\text{m}} \vec v_n\cdot\vec m_c \dx
      \;=\; \langle \vec v_n,\vec m_n\rangle_{\text{lumped}}

   for the nodal coefficients :math:`\vec m_n`, with mass lumping (single-point
   quadrature, ``n_gauss = 1``) so the system is diagonal.

2. **Standard nodal FEM** computation of the field :math:`\vec H_n` from
   :math:`\vec m_n`, exactly as in :ref:`disc_nodal`.

3. **Node → cell projection** of the resulting field, using the dual
   mass-lumped form

   .. math::

      \int_{\Omega_\text{m}} \vec v_c\cdot\vec H_n \dx
      \;=\; \langle \vec v_c,\vec H_c\rangle_{\text{lumped}}.

The energy density is projected the same way (a scalar
:math:`\int v_c\, e_n\,\dx` form). All three projection forms are written as
plain SymPy expressions and handed to the same ``linear_form_cmds`` routine
the rest of NeuralMag uses; you can read the exact code in
``neuralmag/field_terms/field_term.py`` (the ``_generate_code`` classmethod
around lines 124–213).

In a FIC simulation you only see the cell-centred view: ``state.m`` is a
:class:`~neuralmag.VectorCellFunction`, ``state.h_demag`` returns a tensor of
shape ``(n_x, n_y, n_z, 3)``, and so on. The node-side intermediates exist
only inside the generated function.

.. admonition:: When to prefer FIC
   :class: tip

   * Topology optimization, where the design variable :math:`\rho` and the
     material parameters live naturally on cells.
   * Geometries imported from imaging or CAD pipelines that produce
     cell-aligned masks.
   * Interfaces that should remain *sharp* in the discretization rather than
     being smeared by trilinear interpolation of ``m``.

.. _choosing_a_scheme:

Choosing a scheme
-----------------

A short rule of thumb:

==================================================  ==========
Situation                                           Use
==================================================  ==========
Standard problems / smooth fields                   nodal FEM
Topology optimization (cell-based design vars)      FIC
Sharp material interfaces, voxel data               FIC
Comparing against existing FE micromagnetic codes   nodal FEM
==================================================  ==========

If you switch later you only need to change ``state.m`` from a
:class:`~neuralmag.VectorFunction` to a :class:`~neuralmag.VectorCellFunction`
(or vice versa) — every field term re-detects the layout from
``state.m.spaces`` and re-compiles its code on the next access.

Further reading
---------------

* C. Abert, *"Micromagnetics and spintronics: models and numerical
  methods"*, Eur. Phys. J. B **92**, 120 (2019),
  `doi:10.1140/epjb/e2019-90599-6
  <https://doi.org/10.1140/epjb/e2019-90599-6>`_.
* F. Bruckner, S. Koraltan, C. Abert, D. Suess, *"magnum.np: a PyTorch based
  GPU enhanced finite difference micromagnetic simulation framework for
  high level development and inverse design"*, Sci. Rep. **13**, 12054
  (2023).
* C. Abert, F. Bruckner, A. Voronov, M. Lang, S. A. Pathak, S. Holt,
  R. Kraft, R. Allayarov, P. Flauger, S. Koraltan, T. Schrefl, A. Chumak,
  H. Fangohr, D. Suess, *"NeuralMag: an open-source nodal finite-difference
  code for inverse micromagnetics"*, npj Comput. Mater. **11**, 193 (2025),
  `doi:10.1038/s41524-025-01688-1
  <https://doi.org/10.1038/s41524-025-01688-1>`_ — discretization used by
  NeuralMag and reference for the FIC variant.
