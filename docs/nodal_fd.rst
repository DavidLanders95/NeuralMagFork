Nodal Finite-Differences
========================

The nodal finite-difference scheme applies a finite-element approach on a regular cuboidal grid for local field contributions such as the exchange field.
This allows the use of a continuous, piecewise polynomial function space for fields such as the magnetization :math:`\vec{m}` and a piecewise constant function space for material parameters such as the exchange constant :math:`A`, see :numref:`nodal_fd` (a).
For the continuous field, we use product-space 1D-Lagrange (trilinear) basis functions :math:`\phi_i`, see exemplary 2D representation in :numref:`nodal_fd` (b).

For the field computation, we apply the finite-element formalism with mass lumping according to [Abert2019]_, i.e.

.. math::

  H_i &= \sum_j A_{ij} m_j,\\
  A_{ij} &=
  - \left[ \int_{\Omega_\text{m}} \mu_0 M_\text{s} \vec{\phi}_i \cdot \vec{1} \dx \right]^{-1}
  \delta E(\vec{m} = \vec{\phi}_j; \vec{\phi}_i)

with :math:`H_i` and :math:`m_i` being the vectors carrying the :math:`3N` coefficients of the discretized field :math:`\vec{H}` and magnetization :math:`\vec{m}`  respectively and :math:`\delta E(\vec{m} = \vec{\phi}_j; \vec{\phi}_i)` being the functional differential of the Gibbs free energy :math:`E` in the direction :math:`\vec{\phi}_i` evaluated at :math:`\vec{m} = \vec{\phi}_j`.
In case of a quadratic energy contribution such as the exchange field, the operator :math:`A_{ij}` is a constant matrix

.. math::

  A_{ij} =
  - \left[ \int_{\Omega_\text{m}} \mu_0 M_\text{s} \vec{\phi}_i \cdot \vec{1} \dx \right]^{-1}
  A_\text{ex} \nabla \vec{\phi}_i : \nabla \vec{\phi}_j \dx.

Since we apply the finite-element method on a regular grid, the matrix :math:`A` has a regular structure and is fully characterized by the element matrix for a single discretization cell.
Due to this structure, the nodal discretization can be trivially implemented as a matrix-free method with no significant memory nor performance overhead compared to the standard finite-difference methodology.

.. figure:: figures/nodal_fd.png
  :name: nodal_fd
  :scale: 40%

  Nodal finite-difference discretization.
  (a) 1D representation of discretization for continuous field (red) and piecewise constant material parameters (blue).
  (b) 2D basis function :math:`\phi_i(x_1, x_2)` for nodal finite-differences.

For the demagnetization field, we use the well established FFT accelerated fast convolution method employed in standard finite-difference micromagnetics. 
Specifically, we reuse the demagnetization-field implementation of magnum.np, which has been optimized and benchmarked against established codes [Bruckner2023]_.

By combining the fast demagnetization field of finite-difference micromagnetics with the accurate discretization of material interfaces for the remaining field terms, the nodal finite-difference method combines the speed of finite differences with the accuracy of finite elements.

Form Compiler
-------------

NeuralMag features a form compiler that translates arbitrary functionals and linear form in efficient, vectorized PyTorch routines.
As an input the form compiler takes `SymPy <https://www.sympy.org/>`_ expressions with special symbols for the representation of discretized functions and integration measures.
For the representation of discretized fields, NeuralMag introduces the :class:`Variable` class.

NeuralMag uses SymPy for the 

..
    %For the continuous field, we use product-space 1D-Lagrange functions, see Fig.~\ref{fig:nodal}(b) with cell size $\Delta x_1 \Delta x_2 \Delta x_3$ that read
    %\begin{align}
    %    \begin{split}
    %    \phi_{i,j,k}(\vec{x}) = 
    %    &[x_1 + i (\Delta x_1 - 2 x_1)] / \Delta x_1 \cdot\\
    %    &[x_2 + j (\Delta x_2 - 2 x_2)] / \Delta x_2 \cdot\\
    %    &[x_3 + k (\Delta x_3 - 2 x_3)] / \Delta x_3
    %    \end{split}
    %\end{align}
    %with $i,j,k \in \{0,1\}$.

.. [Abert2019] Abert, C. "Micromagnetics and spintronics: models and numerical methods." The European Physical Journal B 92.6 (2019): 1-45.
.. [Bruckner2023] Bruckner, F., Koraltan, S., Abert, C., & Suess, D. "magnum.np: a PyTorch based GPU enhanced finite difference micromagnetic simulation framework for high level development and inverse design." Scientific Reports 13.1 (2023): 12054.
Chicago 
