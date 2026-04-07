Form Compiler
=============

NeuralMag features a form compiler that translates arbitrary functionals and
linear weak forms into efficient, vectorized backend code. As an input it
takes `SymPy <https://www.sympy.org/>`_ expressions written with a small set
of special symbols for discretized functions and integration measures.

.. admonition:: When you need this page
   :class: note

   You normally don't have to read any of this to *use* NeuralMag — the form
   compiler is an implementation detail of the built-in field terms. The
   only common reason to learn it is to **add a new field term** by writing
   its energy as a SymPy expression. Everything else is handled
   automatically.

Function spaces
---------------

NeuralMag supports two function spaces for discretized fields:

* a piecewise polynomial space with degrees of freedom on the **nodes** of the
  cuboid mesh, and
* a piecewise constant space with values defined on the **cells**.

The basis functions :math:`\phi_{ijk}` for the nodal space are defined per
cell as the product of 1D functions in each principal direction:

.. math::

   \phi_{ijk}(x, y, z) = \phi_i^x(x)\,\phi_j^y(y)\,\phi_k^z(z),

where :math:`i,j,k` are the node indices in each direction. The 1D basis
functions :math:`\phi^x_i` are first-order Lagrange polynomials satisfying
the nodal condition

.. math::

   \phi^x_{i}(x_j) = \delta_{ij},

i.e.

.. math::

   \phi^x_{i}(x) = \begin{cases}
     (x - x_{i-1}) / (x_i - x_{i-1}) & \text{if } x_{i-1} \le x \le x_i, \\
     (x - x_{i+1}) / (x_i - x_{i+1}) & \text{if } x_i \le x \le x_{i+1}, \\
     0 & \text{else}.
   \end{cases}

The piecewise-constant basis functions :math:`\theta_{ijk}` are defined on
the cells bounded by the nodes:

.. math::

   \theta_{ijk}(x, y, z) = \theta^x_i(x)\,\theta^y_j(y)\,\theta^z_k(z),

with

.. math::

   \theta^x_i(x) = \begin{cases}
     1 & \text{if } x_i \le x \le x_{i+1}, \\
     0 & \text{else}.
   \end{cases}

NeuralMag supports nodal and cell-based functions in arbitrary dimensions and
even allows mixing of nodal and cell discretization in different principal
directions. For the symbolic representation of discretized fields, the form
compiler exposes :func:`~neuralmag.common.engine.Variable`. For instance, a
scalar field that is nodal in all 3 directions is declared as

.. code:: python

   u = Variable("u", "nnn")

The first argument is the variable name; the second specifies the space in
each direction (``"n"`` for nodal, ``"c"`` for cell). A piecewise constant
3D field would be ``"ccc"``; a mixed space such as ``"ncn"`` yields the
basis function

.. math::

   \kappa_{ijk}(x, y, z) = \phi_i^x(x)\,\theta_j^y(y)\,\phi_k^z(z).

Defining and evaluating functionals
-----------------------------------

The :func:`~neuralmag.common.engine.Variable` constructor returns a SymPy
expression that you can combine with the integration measure ``dV`` to build
arbitrary functionals. The simplest case is integrating a scalar field over
the mesh:

.. code:: python

   form = u * dV()

This form can be turned into efficient backend code via the form compiler:

.. code:: python

   code = functional_code(form)

which produces

.. code:: python

   import torch

   def M(dx, rho, u):
       return (0.125*dx[0]*dx[1]*dx[2]*rho[...]*(
           u[:-1,:-1,:-1] + u[:-1,:-1,1:] + u[:-1,1:,:-1] + u[:-1,1:,1:] +
           u[1:,:-1,:-1]  + u[1:,:-1,1:]  + u[1:,1:,:-1]  + u[1:,1:,1:]
       )).sum()

The generated function takes three tensor arguments:

``dx``
    1D tensor with the cell sizes in each principal direction.

``rho``
    Cell tensor describing the geometry of the integration volume as a
    *material density* (see :doc:`domains`).

``u``
    Nodal tensor with the discretized values of the scalar field.

For a mesh of :math:`10 \times 10 \times 1` cells, ``rho`` has shape
``(10, 10, 1)`` while ``u`` has shape ``(11, 11, 2)`` because nodal fields
have one more entry per direction than cells.

Defining and evaluating linear forms
------------------------------------

The form compiler can also generate code for linear weak forms whose result
is a tensor of dual coefficients. For instance, the action of the mass
matrix :math:`L(v) = \int u\,v\,\dx` on two nodal fields :math:`u` and
:math:`v` is set up as

.. code:: python

   u = Variable("u", "nnn")
   v = Variable("v", "nnn")
   form = u * v * dV()

   code = linear_form_code(form)

By naming convention the variable named ``"v"`` is treated as the test
function. The resulting Python function has the signature

.. code:: python

   def L(result, dx, rho, u):
       result[:-1,:-1,:-1] += dx[0]*dx[1]*dx[2]*rho[...]*( ... )
       result[:-1,:-1,1: ] += dx[0]*dx[1]*dx[2]*rho[...]*( ... )
       ...                                         # 8 stencil contributions

The result is written into the pre-allocated ``result`` tensor, and the
entries of :math:`\vec r` are the linear form evaluated against each basis
vector, :math:`r_i = L(\phi_i)`.

What about bilinear forms?
--------------------------

NeuralMag does **not** generate code for bilinear forms in the sense of
assembling explicit system matrices: with iterative solvers you only need
the action of the operator on a vector, and the linear-form machinery
already provides exactly that. Effective fields, projection forms (see
:doc:`discretization`), and the Gateaux derivatives used by every field term
are all expressed through linear forms.

See also
--------

* :doc:`discretization` — how field terms use the form compiler.
* :doc:`../reference/field_terms` — the built-in field terms and the
  :class:`~neuralmag.FieldTerm` base class.
