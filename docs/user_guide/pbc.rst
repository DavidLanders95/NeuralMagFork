.. _pbc:

Periodic boundary conditions
============================

NeuralMag supports periodic boundary conditions (PBC) per spatial direction.
Enabling PBC lets you simulate systems that are effectively infinite in one
or more directions — thin films, nanowires, stripe domains, or any
representative-volume study — without adding artificial surface charges from
a finite computational box.

The :class:`~neuralmag.Mesh` accepts a ``pbc`` argument that selects one of
three modes in each principal direction:

========================  ============================================
Value (per direction)     Meaning
========================  ============================================
``0`` / ``False``         Open boundary (default).
Positive integer ``N``    *Pseudo*-PBC with ``N`` image copies per side.
``True`` / ``inf``        *True* PBC (periodic kernel, wrap-around).
========================  ============================================

Passing a bare ``True``/``False`` enables or disables PBC in all three
directions at once.

Enabling PBC
------------

.. code-block:: python

   from neuralmag import Mesh

   # Thin-film stripe, periodic in-plane (true PBC)
   mesh = Mesh((64, 64, 4), (5e-9, 5e-9, 5e-9), pbc=(True, True, 0))

   # Long nanowire, pseudo-PBC in x with two image copies per side
   mesh = Mesh((128, 8, 8), (2e-9, 2e-9, 2e-9), pbc=(2, 0, 0))

   # Fully periodic 3D cube
   mesh = Mesh((32, 32, 32), (4e-9,) * 3, pbc=True)

Every state, field term, and solver built on top of such a mesh picks up the
periodic topology automatically — no further API changes are required.

True vs. pseudo-PBC
-------------------

The two periodic modes differ in how the demagnetization convolution is
evaluated:

* **True PBC** uses a *periodic* demagnetization kernel constructed directly
  in Fourier space. The convolution wraps around the computational box
  exactly, so the simulated system is genuinely infinite in the periodic
  directions. NeuralMag follows the construction of
  Bruckner et al. (*Sci. Rep.* **11**, 9202, 2021); see "Further reading"
  below.

* **Pseudo-PBC** keeps the standard open-boundary demagnetization tensor and
  sums the field contributions from ``N`` image copies of the sample on each
  side of the periodic direction. The result converges to the true periodic
  solution as ``N`` grows, at the cost of a larger tensor and slower setup.
  Because it only re-uses the open-boundary kernel, pseudo-PBC works in any
  dimension and with any field term.

In practice: prefer true PBC when it applies (see *Limitations*); use
pseudo-PBC when true PBC is not available or when you want a controllable
approximation with a small ``N``.

Effect on field terms
---------------------

* **Exchange** (both nodal FD and FIC): neighbour stencils wrap around in
  every periodic direction. Both true and pseudo-PBC are supported; the
  exchange operator itself does not distinguish between them.
* **Demagnetization**: the field term detects
  ``all(mesh.pbc[i] == inf)`` at registration time and swaps in the periodic
  kernel (``h_cell_pbc``). For any other ``pbc`` setting, the open-boundary
  kernel is used together with image copies (pseudo-PBC).
* **DMI, anisotropy, external field, interlayer exchange**: local
  contributions — unaffected by the choice of boundary conditions.

Limitations
-----------

* **True PBC for the demagnetization field requires a 3D mesh.** Using
  ``pbc=True`` on a 1D or 2D mesh together with :class:`DemagField` raises a
  ``ValueError``. If you need an effectively 2D setup, use a 3D mesh with a
  single cell in the out-of-plane direction and ``pbc=(True, True, 0)``.
* **Node count differs under PBC.** For a periodic direction ``i``, the
  number of independent nodes is ``n[i]`` instead of ``n[i] + 1``. This is
  handled transparently by :class:`~neuralmag.Function` /
  :class:`~neuralmag.VectorFunction`, but matters when you write a custom
  initial condition that indexes the tensor directly.
* Pseudo-PBC is an approximation controlled by the number of image copies
  ``N``. Convergence should be checked for the problem at hand; increasing
  ``N`` trades accuracy for memory and setup time.

Validation
----------

Reference tests covering PBC live alongside the sources:

* ``tests/unit/pbc_test.py`` — exchange field under true PBC (nodal FD and
  FIC).
* ``tests/unit/demag_field_test.py`` — true-PBC demagnetization on a uniform
  cube (``test_h_cell_pbc_uniform``) and on a periodic stripe geometry
  against the analytical result of Bruckner et al. (2021)
  (``test_h_cell_pbc_stripes``), plus pseudo-PBC geometry tests.

Further reading
---------------

* F. Bruckner, A. Ducevic, P. Heistracher, C. Abert, D. Suess, *"Strayfield
  calculation for micromagnetic simulations using true periodic boundary
  conditions"*, Sci. Rep. **11**, 9202 (2021) — the periodic demagnetization
  kernel used by NeuralMag.

  .. code-block:: bibtex

     @article{bruckner2021strayfield,
       title={Strayfield calculation for micromagnetic simulations using true periodic boundary conditions},
       author={Bruckner, Florian and Ducevic, Amil and Heistracher, Paul and Abert, Claas and Suess, Dieter},
       journal={Scientific Reports},
       volume={11},
       number={1},
       pages={9202},
       year={2021},
       publisher={Nature Publishing Group UK London}
     }

See also
--------

* :doc:`discretization` — nodal FD and FIC schemes that PBC builds on.
* :class:`~neuralmag.Mesh` in the :doc:`../reference/index`.
