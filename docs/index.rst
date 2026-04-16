.. NeuralMag documentation master file, created by
   sphinx-quickstart on Wed Jan 10 17:29:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeuralMag
=========

NeuralMag is a differentiable micromagnetic simulator built around a
`SymPy <https://www.sympy.org/>`_ form compiler, designed specifically with
inverse problems in mind. It runs on top of `JAX
<https://jax.readthedocs.io/en/latest/>`_ or `PyTorch <https://pytorch.org/>`_
and supports CPU and GPU execution out of the box.

What you get
------------

* Nodal finite-difference discretization on a regular cuboid grid, plus a
  cell-centred *FIC* variant that reuses the same form compiler — see
  :doc:`user_guide/discretization`.
* Built-in field contributions: external field, exchange, demagnetization,
  uniaxial / cubic anisotropy, interface and bulk DMI, and interlayer
  exchange.
* A differentiable Landau–Lifshitz–Gilbert time integrator and a small set
  of loggers.
* A *purely functional* state interface: material parameters, applied
  fields, even the geometry mask can be lambdas of other state attributes,
  which makes ``jax.grad`` / ``torch.autograd`` loops natural — see
  :doc:`user_guide/dynamic_attributes`.

Where to go next
----------------

* **New here?** Read :doc:`user_guide/introduction` for the mental model and
  then walk through the :doc:`user_guide/getting_started` notebook.
* **Setting up a multi-material or topology-optimization problem?** See
  :doc:`user_guide/domains` and :doc:`user_guide/dynamic_attributes`.
* **Curious about the math?** :doc:`user_guide/discretization` covers the
  nodal finite-difference and FIC schemes; :doc:`user_guide/form_compiler` explains how
  the SymPy-based form compiler works under the hood.
* **Simulating extended or periodic systems?** See :doc:`user_guide/pbc`
  for true and pseudo-periodic boundary conditions.
* **Looking for an API?** See the :doc:`reference/index`.
* **Looking for a working script?** Browse the :doc:`examples/index`.

Download and install
--------------------

NeuralMag is a Python package and requires Python ≥ 3.8 (≥ 3.10 for the
JAX backend). Install with one of

.. code::

   pip install "neuralmag[jax]"

.. code::

   pip install "neuralmag[torch]"

You can also install both backends and choose between them at runtime via
the ``NM_BACKEND`` environment variable.

How to cite
-----------

If you use NeuralMag in scientific work, please cite the accompanying paper:

   C. Abert, F. Bruckner, A. Voronov, M. Lang, S. A. Pathak, S. Holt,
   R. Kraft, R. Allayarov, P. Flauger, S. Koraltan, T. Schrefl, A. Chumak,
   H. Fangohr, D. Suess, *"NeuralMag: an open-source nodal finite-difference
   code for inverse micromagnetics"*, npj Comput. Mater. **11**, 193 (2025),
   `doi:10.1038/s41524-025-01688-1
   <https://doi.org/10.1038/s41524-025-01688-1>`_.

BibTeX:

.. code-block:: bibtex

   @article{Abert2025NeuralMag,
     author  = {Abert, Claas and Bruckner, Florian and Voronov, Andrii and
                Lang, Martin and Pathak, Swapneel Amit and Holt, Sam and
                Kraft, Roman and Allayarov, Rustam and Flauger, Paul and
                Koraltan, Sabri and Schrefl, Thomas and Chumak, Andrii and
                Fangohr, Hans and Suess, Dieter},
     title   = {{NeuralMag}: an open-source nodal finite-difference code for
                inverse micromagnetics},
     journal = {npj Computational Materials},
     volume  = {11},
     pages   = {193},
     year    = {2025},
     doi     = {10.1038/s41524-025-01688-1},
   }

Contribute
----------

Contributions of any kind — bug fixes, new field terms, documentation
improvements — are very welcome. NeuralMag is licensed under the
`MIT License <https://opensource.org/license/MIT>`_; by contributing you
agree to license your contribution under the same terms.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   reference/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
