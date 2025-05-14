NeuralMag
=========

NeuralMag is a micromagnetic simulation software using the nodal finite-difference
discretization scheme, designed specifically with inverse problems in mind. It uses either
[JAX](https://jax.readthedocs.io/en/latest/) or [PyTorch](https://pytorch.org/) as a
numerical backend for tensor operations and automatic differentiation, enabling
computations on both CPU and GPU systems. At the moment NeuralMag implements the most
common micromagnetic effective-field contributions 

-   external field
-   exchange field
-   demagnetization field
-   uniaxial/cubic anisotropy
-   DMI (interface and bulk)
-   interlayer exchange

as well as a differentiable time-domain solver for the Landau-Lifshitz-Gilbert equation.

NeuralMag is designed in a modular fashion resulting in a very high flexibility for the
problem definition. For instance, all simulation parameters (e.g. material parameters) can
be functions of space, time or any other simulation parameter.

At the heart of NeuralMag is a form compiler powered by [SymPy](https://www.sympy.org/)
that translates arbitrary functionals and linear weak forms into vectorized PyTorch code.
This allows to easily add new effective-field contributions by simply stating the
corresponding energy as a sympy expression.

Documentation
=============

The documentation of NeuralMag including a reference to all classes as well as several
examples can found [here](https://neuralmag.gitlab.io/neuralmag/index.html).

NeuralMag in the cloud
======================

Experience NeuralMag without installing it locally by accessing it directly in the cloud
via Binder. Simply click the badge to get started:
[![Binder](https://notebooks.mpcdf.mpg.de/binder/badge_logo.svg)](https://notebooks.mpcdf.mpg.de/binder/v2/git/https%3A%2F%2Fgitlab.mpcdf.mpg.de%2Fneuralmag%2Fneuralmag.git/binder)

Using NeuralMag on Binder allows you to experience its features, without the hassle
of setting up your local environment. It provides a quick and accessible way to test
and experiment with the software from any device with a web browser.
It is important to note that the Binder-hosted version is a CPU-only JAX
implementation, and it will run slower than a local installation.
Sessions are temporary and may time out after a period of inactivity, and any files
created or modified during your session will not be saved.
To avoid losing your work, please remember to download any files you create or edit
before your session ends.

Download and Install
====================

NeuralMag is a Python package and requires Python \>=3.8 (\>=3.10 for JAX backend). To install the latest version
with pip either run

``` {.sourceCode .}
pip install "neuralmag[jax]"
```

to install NeuralMag with JAX as a backend or

``` {.sourceCode .}
pip install "neuralmag[torch]"
```

to install NeuralMag with PyTorch as a backend. You can also install NeuralMag with both
backends and choose the backend at runtime.


Contribute
==========

Thank you for considering contributing to our project! We welcome any contributions,
whether they are in the form of bug fixes, feature enhancements, documentation improvements,
or any other kind of enhancement. NeuralMag is licensed under the
[MIT License](https://opensource.org/license/MIT). By contributing to this project,
you agree to license your contributions under the terms of the MIT License.
