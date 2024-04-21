.. _getting_started:

Getting Started
===============

As a first example the standard problem #4 [MuMag4]_, proposed by the MuMag group is computed with NeuralMag. This problem describes the switching of a thin strip of permalloy under the influence of an external field.

Since NeuralMag is a Python library, every simulation script is a Python script. In order to use NeuralMag, the Python package ``neuralmag`` needs to by imported in a first step.

.. code:: python

  from neuralmag import *

In the next step, a mesh for the simulation is created. As for the standard finite-difference method, the nodal finite-difference method requires a regular cuboid grid which can be defined by the number of cells and the cell size in the principal directions of the coordinate system.
For the standard problem #4 we define a mesh with :math:`100 \times 24 \times 1` cells with cell size :math:`5 \times 5 \times 3 \text{nm}^3`.

.. code:: python

    mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))

Next, we define a :code:`State` object which manages the state of simulation such as material parameters and the current time and magnetization.

.. code:: python

    state = State(mesh)

Since the standard problem #4 defines a homogeneous material throughout the cuboid sample, the material parameters can be set as simple Python float values

.. code:: python

    state.material.Ms = 8e5
    state.material.A = 1.3e-11
    state.material.alpha = 1.0

The initial magnetization configuration is set homogeneously in (1,1,0) direction in order to ensure relaxation into the desired S-state defined in the MuMag problem.
As opposed to the material parameters, the magnetization is initialied as a :code:`VectorFunction` object which represents a vector field defined on the nodes (vertices) of the cuboid mesh.
We use the :code:`fill` method of the function to object to set the initial magnetization direction on each node.

.. code:: python

    state.m = VectorFunction(state).fill((0.5**0.5, 0.5**0.5, 0))

Next, we set up the individual effective-field contributions for the simulation, namely the exchange field, the demagnetization field and the external field.
Since the first step in the problem definition of the standard problem #4 requires the relaxation of the magnetization at zero external field, we set the initial field strength :code:`h_ext` to zero.
Like the magnetization, the external field is represented as vector function discretized on the nodes of the mesh.
By calling the :code:`register` method on the individual field objects, the state object is extended by routines for the computation of the respective effective field and energy.
In order to compute the commulative effective field including all individual contributions, we initial a :class:`TotalField` object that adds up all contributions and registers routines for the computation of the total effective field with the state object.

.. code:: python

    h_ext = VectorFunction(state).fill((0, 0, 0), expand=True)

    ExchangeField().register(state, "exchange")
    DemagField().register(state, "demag")
    ExternalField(h_ext).register(state, "external")

    TotalField("exchange", "demag", "external").register(state)

Having all material parameters and field contributions in place, we can now proceed to relax the magnetic system into the initial S-state.
We do this, by integrating the Landau-Lifshitz-Gilbert equation for 1 ns.
Since we set the Gilbert damping to :math:`\alpha = 1` this should be sufficient to result in an energetic equilibrium.

.. code:: python

    llg = LLGSolver(state)
    llg.step(1e-9)

The :class:`LLGSolver` object takes all required information for the time integration from the :class:`State` object.
Namely, it calls the :code:`h` method on state that has been registered by the :class:`TotalField` class in order to evaluate the effective field.
After the relaxation to the S-state, we can check the resulting magnetization configuration by write it to a VTI-file.

.. code:: python

    state.write_vti("m", "sstate.vti")

This file can be visualized and analyzed with the open source tool `Paraview <https://www.paraview.org/>`_.
As required by the standard problem #4, in the next step the dynamical switching of the magnetization under the influence of an external field directed slightly tilted to the -x direction is computed.
In oder to simulate this switching process with NeuralMag, we set both the external field and the Gilbert damping constant to the values required by the standard problem.

.. code:: python

    h_ext.fill([-19576.0, 3421.0, 0.0], expand=True)
    state.material.alpha = 0.02
    state.t = 0.0

In a next step we initialize a :class:`Logger` object, that is configured to log the time and the averaged magnetization in a simple CSV file as well as the full magnetization configuration in a series of VTI files in a directory called :code:`data`.
Afterwards, the actual time integration is performed in a loop that calls the :code:`step` function on the LLG solver and the :code:`log` function of the :class:`Logger` object in an alternateing fashion.

.. code:: python

    logger = Logger("data", ["t", "m"], ["m"])
    while state.t < 1e-9:
        logger.log(state)
        llg.step(1e-11)

The complete script reads

.. code:: python

    from neuralmag import *

    # setup mesh and state
    mesh = Mesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
    state = State(mesh)

    # setup material and m0
    state.material.Ms = 8e5
    state.material.A = 1.3e-11
    state.material.alpha = 1.0

    # initialize nodal vector functions for magneization and external field
    state.m = VectorFunction(state).fill((0.5**0.5, 0.5**0.5, 0))
    h_ext = VectorFunction(state).fill((0, 0, 0), expand=True)

    # register effective field contributions
    ExchangeField().register(state, "exchange")
    DemagField().register(state, "demag")
    ExternalField(h_ext).register(state, "external")
    TotalField("exchange", "demag", "external").register(state)

    # relax to s-state
    llg = LLGSolver(state)
    llg.step(1e-9)

    state.write_vti("m", "sstate.vti")

    # set external field and damping to perform switch
    h_ext.fill([-19576.0, 3421.0, 0.0], expand=True)
    state.material.alpha = 0.02
    state.t = 0.0

    logger = Logger("data", ["t", "m"], ["m"])
    while state.t < 1e-9:
        logger.log(state)
        llg.step(1e-11)
