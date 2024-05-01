NeuralMag
=========

NeuralMag is a micromagnetic simulation software using the nodal
finite-difference discretization scheme, designed specifically with
inverse problems in mind. It uses [PyTorch](https://pytorch.org/) as a
numerical backend for tensor operations and automatic differentiation,
enabling computations on both CPU and GPU systems. At the moment
NeuralMag implements the most common micromagnetic effective-field
contributions

-   external field
-   exchange field
-   demagnetization field
-   uniaxial anisotropy
-   DMI (interface and bulk)
-   interlayer exchange

as well as a differentiable time-domain solver for the
Landau-Lifshitz-Gilbert equation.

NeuralMag is designed in a modular fashion resulting in a very high
flexibility for the problem definition. For instance, all simulation
parameters (e.g. material parameters) can be functions of space, time or
any other simulation parameter.

At the heart of NeuralMag is a form compiler powered by
[SymPy](https://www.sympy.org/) that translates arbitrary functionals
and linear weak forms into vectorized PyTorch code. This allows to
easily add new effective-field contributions by simply stating the
corresponding energy as a sympy expression.

Documentation
=============

The documentation of NeuralMag including a reference to all classes as
well as several examples can found [here](https://neuralmag.gitlab.io/neuralmag/index.html).


Download and Install
====================

NeuralMag is a Python package and requires Python \>=3.8. To install the
latest version from Gitlab with pip run

``` {.sourceCode .}
pip install git+https://gitlab.com/neuralmag/neuralmag.git
```

Contribute
==========

Thank you for considering contributing to our project! We welcome any
contributions, whether they are in the form of bug fixes, feature
enhancements, documentation improvements, or any other kind of
enhancement. NeuralMag is licensed under the [GNU Lesser General Public
License (LPGL)](https://www.gnu.org/licenses/). By contributing to this
project, you agree to license your contributions under the terms of the
LGPL.
